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
#line 18

#define CAMERA_FAR

in vec2 uv;

layout(binding = 0) uniform sampler2D tex_last;
layout(binding = 1) uniform sampler2D tex_current;
layout(binding = 2) uniform sampler2D tex_motion_vectors;
layout(binding = 3) uniform sampler2D tex_current_depth;

layout(location = 0) out vec3 clean_color;

void main() {
    float current_depth = texture(tex_current_depth, uv).r;
    vec3 current_color = texture(tex_current, uv).rgb;
    if (current_depth >= CAMERA_FAR) {
        clean_color = current_color;
        return;
    }

    vec2 motion_vector = texture(tex_motion_vectors, closest_fragment(tex_current_depth, uv)).rg;
    vec2 old_uv = uv + motion_vector;
    // reject history
    if (any(lessThan(old_uv, vec2(0))) || any(greaterThan(old_uv, vec2(1)))) {
        clean_color = current_color;
        return;
    }
    vec3 history_color = texture(tex_last, old_uv).rgb;
    vec3 min_color = current_color;
    vec3 max_color = current_color;
    min_max_color_neighborhood(tex_current, uv, min_color, max_color);
    vec3 history_color_clamped = clamp(history_color, min_color, max_color);
    clean_color = mix(current_color, history_color_clamped, 0.7);
}
#endif
