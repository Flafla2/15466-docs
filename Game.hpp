#pragma once

#include "GL.hpp"

#include <SDL.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <random>
#include <vector>

// The 'Game' struct holds all of the game-relevant state,
// and is called by the main loop.

struct Game {
	//Game creates OpenGL resources (i.e. vertex buffer objects) in its
	//constructor and frees them in its destructor.
	Game(SDL_Window *window);
	~Game();

	//handle_event is called when new mouse or keyboard events are received:
	// (note that this might be many times per frame or never)
	//The function should return 'true' if it handled the event.
	bool handle_event(SDL_Event const &evt, glm::uvec2 window_size);

	//update is called at the start of a new frame, after events are handled:
	void update(float elapsed);

	//draw is called after update:
	void draw(glm::uvec2 drawable_size);

	//------- opengl resources -------

	//shader program that draws lit objects with vertex colors:
	struct {
		GLuint program = -1U; //program object

		//uniform locations:
		GLuint object_to_clip_mat4 = -1U;
		GLuint object_to_light_mat4x3 = -1U;
		GLuint normal_to_light_mat3 = -1U;
		GLuint sun_direction_vec3 = -1U;
		GLuint sun_color_vec3 = -1U;
		GLuint sky_direction_vec3 = -1U;
		GLuint sky_color_vec3 = -1U;

		//attribute locations:
		GLuint Position_vec4 = -1U;
		GLuint Normal_vec3 = -1U;
		GLuint Color_vec4 = -1U;
	} simple_shading;

	//mesh data, stored in a vertex buffer:
	GLuint meshes_vbo = -1U; //vertex buffer holding mesh data

	//The location of each mesh in the meshes vertex buffer:
	struct Mesh {
		GLint first = 0;
		GLsizei count = 0;
	};

	Mesh paddle_mesh;
	Mesh track_mesh;
	Mesh ball_mesh;
	Mesh lose_text_mesh;
	Mesh coin_mesh;
	Mesh bomb_mesh;

	SDL_Window *window;

	GLuint meshes_for_simple_shading_vao = -1U; //vertex array object that describes how to connect the meshes_vbo to the simple_shading_program

	//------- game state -------

	std::mt19937 mt = std::mt19937(0xbead1234);

	const double track_size = 4.7;
	const double paddle_size = 2.0;
	const double move_speed = 0.75;
	const double ball_speed = 3.0;

	// Min: 0.0, Max: 1.0
	// Current position of the paddle.  0.0/1.0 is 3-o'clock position going CW
	double paddle_pos = 0;
	glm::vec2 ball_vel = glm::vec2(1,0);
	glm::vec2 ball_pos = glm::vec2(0,0);
	glm::uvec2 camera_dim = glm::uvec2(6,6);

	enum game_state {
		PRE_GAME, IN_GAME, POST_GAME
	};

	std::chrono::steady_clock::time_point start_time;

	game_state cur_state = PRE_GAME;
	double state_timer = 0.0;

	struct coin_info {
		glm::vec2 pos;
		float fade_time;
		float rot;
	};

	std::vector<coin_info> coins;
	double next_coin_spawn_time = 0.0;

	uint score = 0;

	struct {
		bool move_cw = false;
		bool move_ccw = false;
	} controls;

};
