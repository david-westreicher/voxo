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

in vec2 uv;

layout(binding = 0) uniform sampler2D tex_last;
layout(binding = 1) uniform sampler2D tex_current;
layout(binding = 2) uniform sampler2D tex_motion_vectors;
layout(binding = 3) uniform sampler2D tex_last_depth;
layout(binding = 4) uniform sampler2D tex_last_normals;
layout(binding = 5) uniform sampler2D tex_current_depth;
layout(binding = 6) uniform sampler2D tex_current_normals;

layout(location = 0) out vec3 clean_color;

vec2 texel_size = 1.0 / vec2(textureSize(tex_current, 0));

bool reject_history(float last_depth, float current_depth, vec3 last_normal, vec3 current_normal) {
    // Depth rejection (relative + absolute)
    float depth_diff = abs(last_depth - current_depth);

    float depth_threshold = 1.0;

    if (depth_diff > depth_threshold) {
        return true;
    }

    // Normal rejection
    float normal_similarity = dot(last_normal, current_normal);

    if (normal_similarity < 0.9)
        return true;

    return false;
}
vec3 clamp_history_color(sampler2D current_color_tex, vec2 uv, vec3 prev_color) {
    ivec2 size = textureSize(current_color_tex, 0);
    ivec2 pixel = ivec2(uv * SCREEN_DIMENSIONS);

    vec3 box_min = vec3(1e30);
    vec3 box_max = vec3(-1e30);

    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            ivec2 p = pixel + ivec2(x, y);
            vec3 c = texelFetch(current_color_tex, p, 0).rgb;

            box_min = min(box_min, c);
            box_max = max(box_max, c);
        }
    }

    return clamp(prev_color, box_min, box_max);
}

void main() {
    ivec2 last_frame_uv = ivec2(floor((uv + texture(tex_motion_vectors, uv).rg) * SCREEN_DIMENSIONS));
    vec3 current = texture(tex_current, uv).rgb;

    bool outside = any(lessThanEqual(last_frame_uv, ivec2(5))) || any(greaterThanEqual(last_frame_uv, SCREEN_DIMENSIONS - ivec2(5)));
    if (outside) {
        clean_color = current;
        return;
    }

    float last_depth = texelFetch(tex_last_depth, last_frame_uv, 0).r;
    vec3 last_normal = decodeNormalRGB10A2(texelFetch(tex_last_normals, last_frame_uv, 0).rgb);
    float current_depth = texture(tex_current_depth, uv).r;
    vec3 current_normal = decodeNormalRGB10A2(texture(tex_current_normals, uv).rgb);

    if (current_depth > 100) {
        clean_color = vec3(0);
        return;
    }
    if (reject_history(last_depth, current_depth, last_normal, current_normal)) {
        clean_color = current;
    } else {
        vec3 last = texelFetch(tex_last, last_frame_uv, 0).rgb;
        vec3 last_clamped = clamp_history_color(tex_current, uv, last);
        clean_color = mix(current, last_clamped, 0.9);
    }
}
#endif
