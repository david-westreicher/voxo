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

    float depth_threshold = 3.0;

    if (depth_diff > depth_threshold)
        return true;

    // Normal rejection
    float normal_similarity = dot(last_normal, current_normal);

    if (normal_similarity < 0.9)
        return true;

    return false;
}

void main() {
    vec2 last_frame_uv = uv + texture(tex_motion_vectors, uv).rg;
    vec3 current = texture(tex_current, uv).rgb;

    bool outside = any(lessThanEqual(last_frame_uv, texel_size * 2.0)) || any(greaterThanEqual(last_frame_uv, vec2(1.0) - texel_size * 2.0));
    if (outside) {
        clean_color = current;
        return;
    }

    float last_depth = texture(tex_last_depth, last_frame_uv).r;
    vec3 last_normal = decodeNormalRGB10A2(texture(tex_last_normals, last_frame_uv).rgb);
    float current_depth = texture(tex_current_depth, last_frame_uv).r;
    vec3 current_normal = decodeNormalRGB10A2(texture(tex_current_normals, uv).rgb);

    if (reject_history(last_depth, current_depth, last_normal, current_normal)) {
        clean_color = current;
    } else {
        vec3 last = texture(tex_last, last_frame_uv).rgb;
        clean_color = mix(current, last, 0.9);
    }
}
#endif
