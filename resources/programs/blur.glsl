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
#line 18

#define HORIZONTAL 0

in vec2 uv;

layout(binding = 0) uniform sampler2D u_input_texture;

layout(location = 0) out vec3 fragColor;

vec2 texel_size = 1.0 / textureSize(u_input_texture, 0);

void main() {
    vec3 result = vec3(0.0);

    #if HORIZONTAL == 1
    vec2 direction = vec2(texel_size.x, 0.0);
    #else
    vec2 direction = vec2(0.0, texel_size.y);
    #endif

    result += texture(u_input_texture, uv - direction * 3.5).rgb * 0.05;
    result += texture(u_input_texture, uv - direction * 1.5).rgb * 0.25;
    result += texture(u_input_texture, uv).rgb * 0.40;
    result += texture(u_input_texture, uv + direction * 1.5).rgb * 0.25;
    result += texture(u_input_texture, uv + direction * 3.5).rgb * 0.05;

    fragColor = result;
}
#endif
