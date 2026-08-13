#version 420

#if defined VERTEX_SHADER

in vec3 in_position;
in vec2 in_texcoord_0;

out vec2 uv;

void main() {
    gl_Position = vec4(in_position, 1);
    uv = in_texcoord_0;
}

#elif defined FRAGMENT_SHADER
#include programs/utils.glsl
#line 18 3

in vec2 uv;

layout(binding = 0) uniform sampler2D u_image;
layout(binding = 1) uniform sampler2D u_last_image;
layout(binding = 2) uniform sampler2D u_motion_vectors;

uniform mat4 u_view;
uniform mat4 u_proj;

layout(location = 0) out vec3 fragColor;

vec2 texel_size = 1 / textureSize(u_image, 0).xy;

void color_clamp(sampler2D u_image, vec2 uv, inout vec3 min_color, inout vec3 max_color) {
    for (int x = -2; x <= 2; ++x) {
        for (int y = -2; y <= 2; ++y) {
            vec3 tex_sample = texture(u_image, uv + vec2(x, y) * texel_size).rgb;
            min_color = min(min_color, tex_sample);
            max_color = max(max_color, tex_sample);
        }
    }
}

void main() {
    vec2 motion_vector = texture(u_motion_vectors, uv).rg;
    vec3 current_color = texture(u_image, uv).rgb;
    if (motion_vector.x == -2.0) {
        fragColor = current_color;
        return;
    }
    vec2 old_uv = uv + motion_vector;
    if (any(lessThan(old_uv, vec2(0))) || any(greaterThan(old_uv, vec2(1)))) {
        fragColor = current_color;
        return;
    }
    vec3 last_color = texture(u_last_image, old_uv).rgb;
    if (length(motion_vector * SCREEN_DIMENSIONS) < 0.01) {
        fragColor = mix(current_color, last_color, 0.95);
        return;
    }

    vec3 curr_min_col = current_color;
    vec3 curr_max_col = current_color;
    color_clamp(u_image, uv, curr_min_col, curr_max_col);
    vec3 last_color_clamped = clamp(last_color, curr_min_col, curr_max_col);

    vec3 last_min_col = last_color;
    vec3 last_max_col = last_color;
    color_clamp(u_last_image, old_uv, last_min_col, last_max_col);

    float history_diff = distance(curr_max_col, last_max_col) + distance(curr_min_col, last_min_col);
    float trust_history = clamp(history_diff * 0.2, 0.0, 1.0);
    vec3 last_color_trust = mix(last_color_clamped, last_color, trust_history);
    fragColor = mix(current_color, last_color_trust, 0.6);
}
#endif
