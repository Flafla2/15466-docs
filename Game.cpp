#include "Game.hpp"

#include "gl_errors.hpp" //helper for dumpping OpenGL error messages
#include "read_chunk.hpp" //helper for reading a vector of structures from a file
#include "data_path.hpp" //helper to get paths relative to executable

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>

#include <iostream>
#include <fstream>
#include <map>
#include <cstddef>

//helper defined later; throws if shader compilation fails:
static GLuint compile_shader(GLenum type, std::string const &source);

Game::Game(SDL_Window *window) {
	{ //create an opengl program to perform sun/sky (well, directional+hemispherical) lighting:
		GLuint vertex_shader = compile_shader(GL_VERTEX_SHADER,
			"#version 330\n"
			"uniform mat4 object_to_clip;\n"
			"uniform mat4x3 object_to_light;\n"
			"uniform mat3 normal_to_light;\n"
			"layout(location=0) in vec4 Position;\n" //note: layout keyword used to make sure that the location-0 attribute is always bound to something
			"in vec3 Normal;\n"
			"in vec4 Color;\n"
			"out vec3 position;\n"
			"out vec3 normal;\n"
			"out vec4 color;\n"
			"void main() {\n"
			"	gl_Position = object_to_clip * Position;\n"
			"	position = object_to_light * Position;\n"
			"	normal = normal_to_light * Normal;\n"
			"	color = Color;\n"
			"}\n"
		);

		GLuint fragment_shader = compile_shader(GL_FRAGMENT_SHADER,
			"#version 330\n"
			"uniform vec3 sun_direction;\n"
			"uniform vec3 sun_color;\n"
			"uniform vec3 sky_direction;\n"
			"uniform vec3 sky_color;\n"
			"in vec3 position;\n"
			"in vec3 normal;\n"
			"in vec4 color;\n"
			"out vec4 fragColor;\n"
			"void main() {\n"
			"	vec3 total_light = vec3(0.0, 0.0, 0.0);\n"
			"	vec3 n = normalize(normal);\n"
			"	{ //sky (hemisphere) light:\n"
			"		vec3 l = sky_direction;\n"
			"		float nl = 0.5 + 0.5 * dot(n,l);\n"
			"		total_light += nl * sky_color;\n"
			"	}\n"
			"	{ //sun (directional) light:\n"
			"		vec3 l = sun_direction;\n"
			"		float nl = max(0.0, dot(n,l));\n"
			"		total_light += nl * sun_color;\n"
			"	}\n"
			"	fragColor = vec4(color.rgb * total_light, color.a);\n"
			"}\n"
		);

		this->window = window;

		simple_shading.program = glCreateProgram();
		glAttachShader(simple_shading.program, vertex_shader);
		glAttachShader(simple_shading.program, fragment_shader);
		//shaders are reference counted so this makes sure they are freed after program is deleted:
		glDeleteShader(vertex_shader);
		glDeleteShader(fragment_shader);

		//link the shader program and throw errors if linking fails:
		glLinkProgram(simple_shading.program);
		GLint link_status = GL_FALSE;
		glGetProgramiv(simple_shading.program, GL_LINK_STATUS, &link_status);
		if (link_status != GL_TRUE) {
			std::cerr << "Failed to link shader program." << std::endl;
			GLint info_log_length = 0;
			glGetProgramiv(simple_shading.program, GL_INFO_LOG_LENGTH, &info_log_length);
			std::vector< GLchar > info_log(info_log_length, 0);
			GLsizei length = 0;
			glGetProgramInfoLog(simple_shading.program, GLsizei(info_log.size()), &length, &info_log[0]);
			std::cerr << "Info log: " << std::string(info_log.begin(), info_log.begin() + length);
			throw std::runtime_error("failed to link program");
		}
	}

	{ //read back uniform and attribute locations from the shader program:
		simple_shading.object_to_clip_mat4 = glGetUniformLocation(simple_shading.program, "object_to_clip");
		simple_shading.object_to_light_mat4x3 = glGetUniformLocation(simple_shading.program, "object_to_light");
		simple_shading.normal_to_light_mat3 = glGetUniformLocation(simple_shading.program, "normal_to_light");

		simple_shading.sun_direction_vec3 = glGetUniformLocation(simple_shading.program, "sun_direction");
		simple_shading.sun_color_vec3 = glGetUniformLocation(simple_shading.program, "sun_color");
		simple_shading.sky_direction_vec3 = glGetUniformLocation(simple_shading.program, "sky_direction");
		simple_shading.sky_color_vec3 = glGetUniformLocation(simple_shading.program, "sky_color");

		simple_shading.Position_vec4 = glGetAttribLocation(simple_shading.program, "Position");
		simple_shading.Normal_vec3 = glGetAttribLocation(simple_shading.program, "Normal");
		simple_shading.Color_vec4 = glGetAttribLocation(simple_shading.program, "Color");
	}

	struct Vertex {
		glm::vec3 Position;
		glm::vec3 Normal;
		glm::u8vec4 Color;
	};
	static_assert(sizeof(Vertex) == 28, "Vertex should be packed.");

	{ //load mesh data from a binary blob:
		std::ifstream blob(data_path("meshes.blob"), std::ios::binary);
		//The blob will be made up of three chunks:
		// the first chunk will be vertex data (interleaved position/normal/color)
		// the second chunk will be characters
		// the third chunk will be an index, mapping a name (range of characters) to a mesh (range of vertex data)

		//read vertex data:
		std::vector< Vertex > vertices;
		read_chunk(blob, "dat0", &vertices);

		//read character data (for names):
		std::vector< char > names;
		read_chunk(blob, "str0", &names);

		//read index:
		struct IndexEntry {
			uint32_t name_begin;
			uint32_t name_end;
			uint32_t vertex_begin;
			uint32_t vertex_end;
		};
		static_assert(sizeof(IndexEntry) == 16, "IndexEntry should be packed.");

		std::vector< IndexEntry > index_entries;
		read_chunk(blob, "idx0", &index_entries);

		if (blob.peek() != EOF) {
			std::cerr << "WARNING: trailing data in meshes file." << std::endl;
		}

		//upload vertex data to the graphics card:
		glGenBuffers(1, &meshes_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, meshes_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * vertices.size(), vertices.data(), GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		//create map to store index entries:
		std::map< std::string, Mesh > index;
		for (IndexEntry const &e : index_entries) {
			if (e.name_begin > e.name_end || e.name_end > names.size()) {
				throw std::runtime_error("invalid name indices in index.");
			}
			if (e.vertex_begin > e.vertex_end || e.vertex_end > vertices.size()) {
				throw std::runtime_error("invalid vertex indices in index.");
			}
			Mesh mesh;
			mesh.first = e.vertex_begin;
			mesh.count = e.vertex_end - e.vertex_begin;
			auto ret = index.insert(std::make_pair(
				std::string(names.begin() + e.name_begin, names.begin() + e.name_end),
				mesh));
			if (!ret.second) {
				throw std::runtime_error("duplicate name in index.");
			}
		}

		//look up into index map to extract meshes:
		auto lookup = [&index](std::string const &name) -> Mesh {
			auto f = index.find(name);
			if (f == index.end()) {
				throw std::runtime_error("Mesh named '" + name + "' does not appear in index.");
			}
			return f->second;
		};
		track_mesh = lookup("track");
		ball_mesh = lookup("Sphere");
		paddle_mesh = lookup("Cube");
		lose_text_mesh = lookup("losertext");
		coin_mesh = lookup("coin");
		bomb_mesh = lookup("bomb");
	}

	{ //create vertex array object to hold the map from the mesh vertex buffer to shader program attributes:
		glGenVertexArrays(1, &meshes_for_simple_shading_vao);
		glBindVertexArray(meshes_for_simple_shading_vao);
		glBindBuffer(GL_ARRAY_BUFFER, meshes_vbo);
		//note that I'm specifying a 3-vector for a 4-vector attribute here, and this is okay to do:
		glVertexAttribPointer(simple_shading.Position_vec4, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLbyte *)0 + offsetof(Vertex, Position));
		glEnableVertexAttribArray(simple_shading.Position_vec4);
		if (simple_shading.Normal_vec3 != -1U) {
			glVertexAttribPointer(simple_shading.Normal_vec3, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLbyte *)0 + offsetof(Vertex, Normal));
			glEnableVertexAttribArray(simple_shading.Normal_vec3);
		}
		if (simple_shading.Color_vec4 != -1U) {
			glVertexAttribPointer(simple_shading.Color_vec4, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(Vertex), (GLbyte *)0 + offsetof(Vertex, Color));
			glEnableVertexAttribArray(simple_shading.Color_vec4);
		}
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	GL_ERRORS();

	ball_pos = glm::vec3(0, 0, 0);
	start_time = std::chrono::high_resolution_clock::now();

	cur_state = PRE_GAME;
	state_timer = 3.0f;
}

Game::~Game() {
	glDeleteVertexArrays(1, &meshes_for_simple_shading_vao);
	meshes_for_simple_shading_vao = -1U;

	glDeleteBuffers(1, &meshes_vbo);
	meshes_vbo = -1U;

	glDeleteProgram(simple_shading.program);
	simple_shading.program = -1U;

	GL_ERRORS();
}

bool Game::handle_event(SDL_Event const &evt, glm::uvec2 window_size) {
	//ignore any keys that are the result of automatic key repeat:
	if (evt.type == SDL_KEYDOWN && evt.key.repeat) {
		return false;
	}
	//movement controls
	if (evt.type == SDL_KEYDOWN || evt.type == SDL_KEYUP) {
		if (evt.key.keysym.scancode == SDL_SCANCODE_Q) {
			controls.move_ccw = (evt.type == SDL_KEYDOWN);
			return true;
		} else if (evt.key.keysym.scancode == SDL_SCANCODE_E) {
			controls.move_cw = (evt.type == SDL_KEYDOWN);
			return true;
		}
	}
	return false;
}

float cross2(glm::vec2 v, glm::vec2 w) {
	return (v.x*w.y) - (v.y*w.x);
}

bool ray_line_isect(glm::vec2 ro, glm::vec2 rd, glm::vec2 p0, glm::vec2 p1, double *out) {
	// Adapted from this explanation
	// https://rootllama.wordpress.com/2014/06/20/ray-line-segment-intersection-test-in-2d/

	glm::vec2 v1 = ro - p0;
	glm::vec2 v2 = p1 - p0;
	glm::vec2 v3(-rd.y, rd.x);

	double d = glm::dot(v2, v3);

	if(d < 0.001)
		return false;

	*out = abs(cross2(v2, v1)) / d;

	return true;
}

glm::vec2 reflect_ray(glm::vec2 ray, glm::vec2 norm) {
	// project ray onto norm of reflection
	glm::vec2 pr = glm::dot(ray, norm) / glm::dot(norm, norm) * norm;
	// subtract projection from ray
	glm::vec2 ret = ray - pr;
	// negate what's left (the reflection)
	pr *= -1;
	// add the projection back (that part wasn't reflected)
	return ret + pr;
}

void Game::update(float elapsed) {
	auto cur_time = std::chrono::high_resolution_clock::now();
	float game_time = std::chrono::duration<float>(cur_time - start_time).count();

	if (controls.move_cw) {
		paddle_pos += move_speed * elapsed;
	}
	if (controls.move_ccw) {
		paddle_pos -= move_speed * elapsed;
	}
	paddle_pos = fmod(1.0 + paddle_pos, 1.0);

	if(cur_state == PRE_GAME) {
		ball_pos = glm::vec3(0, 0, 0);

		if(game_time > state_timer) {
			ball_vel.x = (double)(mt() % 1000)/1000.0;
			ball_vel.y = (double)(mt() % 1000)/1000.0;
			ball_vel = glm::normalize(ball_vel) * (float)ball_speed;

			cur_state = IN_GAME;
		}
		return;
	} else if(cur_state == POST_GAME) {
		if(game_time > state_timer) {
			cur_state = PRE_GAME;
			state_timer = game_time + 3.0;
		}
		return;
	}

	ball_pos += ball_vel * elapsed;

	for(int x = 0; x < coins.size(); x++) {
		coins[x].rot += elapsed * 10.0;
		if(glm::length(coins[x].pos - ball_pos) < 0.2) {
			score++; 
			coins.erase(coins.begin() + x);
			x--; // avoids skipping
		} else if(coins[x].fade_time > game_time) {
			coins.erase(coins.begin() + x);
			x--;
		}
	}
	if(game_time >= next_coin_spawn_time) {
		std::cout << "coin" << std::endl;
		next_coin_spawn_time = game_time + (double)(mt() % 1000)/1000.0 * 10.0;
		coin_info nc;
		nc.pos.x = (double)(mt() % 1000)/1000.0;
		nc.pos.y = (double)(mt() % 1000)/1000.0;
		nc.pos = glm::normalize(nc.pos) * 3.0f;
		nc.rot = (double)(mt() % 1000)/1000.0;
		nc.fade_time = game_time + (double)(mt() % 1000)/1000.0 * 5.0;
	}

	auto paddle_angle = 2.0 * glm::pi<double>() * paddle_pos;

	auto ball_vel_n = glm::normalize(ball_vel);
	auto paddle_wpos = glm::vec2(cos(paddle_angle), sin(paddle_angle)) * (float)(track_size / 2);
	auto paddle_axis = glm::normalize(glm::vec2(-paddle_wpos.y, paddle_wpos.x)) * (float)paddle_size;

	auto paddle_p1 = paddle_wpos - paddle_axis * 0.5f;
	auto paddle_p2 = paddle_wpos + paddle_axis * 0.5f;

	double dist = 0;
	bool hit = ray_line_isect(ball_pos, ball_vel_n, paddle_p1, paddle_p2, &dist);

	if(hit && glm::dot(ball_vel_n, paddle_wpos) >= 0 && dist <= ball_vel.length() * elapsed + 0.2f) {
		ball_vel = reflect_ray(ball_vel, glm::normalize(paddle_wpos));
		score++;
		std::cout << "hit" << std::endl;
	}

	if(glm::dot(ball_pos, ball_pos) > track_size*track_size*0.25) {
		cur_state = POST_GAME;
		state_timer = game_time + 2.0;
		std::cout << "Lose Condition" << std::endl;

		char scorebuf[512];
		snprintf(scorebuf, 512, "Score: %d", (int)score);
		SDL_ShowSimpleMessageBox(0, "You Lost", scorebuf, window);
		score = 0;
	}
}

void Game::draw(glm::uvec2 drawable_size) {
	//Set up a transformation matrix to fit the board in the window:
	glm::mat4 world_to_clip;
	{
		float aspect = float(drawable_size.x) / float(drawable_size.y);

		//want scale such that board * scale fits in [-aspect,aspect]x[-1.0,1.0] screen box:
		float scale = glm::min(
			2.0f * aspect / float(camera_dim.x),
			2.0f / float(camera_dim.y)
		);

		//center of board will be placed at center of screen:
		glm::vec2 center = 0.5f * glm::vec2(camera_dim);

		//NOTE: glm matrices are specified in column-major order
		world_to_clip = glm::mat4(
			scale / aspect, 0.0f, 0.0f, 0.0f,
			0.0f, scale, 0.0f, 0.0f,
			0.0f, 0.0f,-1.0f, 0.0f,
			-(scale / aspect) * center.x, -scale * center.y, 0.0f, 1.0f
		);
	}

	//set up graphics pipeline to use data from the meshes and the simple shading program:
	glBindVertexArray(meshes_for_simple_shading_vao);
	glUseProgram(simple_shading.program);

	glUniform3fv(simple_shading.sun_color_vec3, 1, glm::value_ptr(glm::vec3(0.81f, 0.81f, 0.76f)));
	glUniform3fv(simple_shading.sun_direction_vec3, 1, glm::value_ptr(glm::normalize(glm::vec3(-0.2f, 0.2f, 1.0f))));
	glUniform3fv(simple_shading.sky_color_vec3, 1, glm::value_ptr(glm::vec3(0.2f, 0.2f, 0.3f)));
	glUniform3fv(simple_shading.sky_direction_vec3, 1, glm::value_ptr(glm::vec3(0.0f, 1.0f, 0.0f)));

	//helper function to draw a given mesh with a given transformation:
	auto draw_mesh = [&](Mesh const &mesh, glm::mat4 const &object_to_world) {
		//set up the matrix uniforms:
		if (simple_shading.object_to_clip_mat4 != -1U) {
			glm::mat4 object_to_clip = world_to_clip * object_to_world;
			glUniformMatrix4fv(simple_shading.object_to_clip_mat4, 1, GL_FALSE, glm::value_ptr(object_to_clip));
		}
		if (simple_shading.object_to_light_mat4x3 != -1U) {
			glUniformMatrix4x3fv(simple_shading.object_to_light_mat4x3, 1, GL_FALSE, glm::value_ptr(object_to_world));
		}
		if (simple_shading.normal_to_light_mat3 != -1U) {
			//NOTE: if there isn't any non-uniform scaling in the object_to_world matrix, then the inverse transpose is the matrix itself, and computing it wastes some CPU time:
			glm::mat3 normal_to_world = glm::inverse(glm::transpose(glm::mat3(object_to_world)));
			glUniformMatrix3fv(simple_shading.normal_to_light_mat3, 1, GL_FALSE, glm::value_ptr(normal_to_world));
		}

		//draw the mesh:
		glDrawArrays(GL_TRIANGLES, mesh.first, mesh.count);
	};

	// draw track at origin
	glm::mat4 tran(1), rot(1), identity(1);
	glm::vec3 game_center = glm::vec3(camera_dim.x, camera_dim.y, 0) * 0.5f;
	tran = glm::translate(identity, game_center);
	draw_mesh(track_mesh, tran);

	// draw paddle
	double angle = 2.0 * glm::pi<double>() * paddle_pos;
	
	glm::vec2 posn = glm::vec2(cos(angle), sin(angle)) * (float)(track_size / 2);
	tran = glm::translate(identity, game_center + glm::vec3(posn.x, posn.y, 0));
	rot = glm::rotate(identity, (glm::mediump_float)(glm::pi<double>()/2 + angle), glm::vec3(0, 0, 1));
	draw_mesh(paddle_mesh, tran * rot);

	// draw ball
	tran = glm::translate(identity, glm::vec3(ball_pos.x, ball_pos.y, 0) + game_center);
	draw_mesh(ball_mesh, tran);

	// draw coins
	for(int x = 0; x < coins.size(); x++) {
		tran = glm::translate(identity, glm::vec3(coins[x].pos.x, coins[x].pos.y, 0) + game_center);
		rot = glm::rotate(identity, coins[x].rot, glm::vec3(0,0,1));
		draw_mesh(coin_mesh, tran * rot);
	}

	if(cur_state == POST_GAME) {
		// the player just lost, draw the "you lose" text
		tran = glm::translate(identity, glm::vec3(0, 1, 0) + game_center);
		draw_mesh(lose_text_mesh, tran);
	}

	glUseProgram(0);

	GL_ERRORS();
}



//create and return an OpenGL vertex shader from source:
static GLuint compile_shader(GLenum type, std::string const &source) {
	GLuint shader = glCreateShader(type);
	GLchar const *str = source.c_str();
	GLint length = GLint(source.size());
	glShaderSource(shader, 1, &str, &length);
	glCompileShader(shader);
	GLint compile_status = GL_FALSE;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &compile_status);
	if (compile_status != GL_TRUE) {
		std::cerr << "Failed to compile shader." << std::endl;
		GLint info_log_length = 0;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &info_log_length);
		std::vector< GLchar > info_log(info_log_length, 0);
		GLsizei length = 0;
		glGetShaderInfoLog(shader, GLsizei(info_log.size()), &length, &info_log[0]);
		std::cerr << "Info log: " << std::string(info_log.begin(), info_log.begin() + length);
		glDeleteShader(shader);
		throw std::runtime_error("Failed to compile shader.");
	}
	return shader;
}
