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

in vec2 uv;

layout(binding = 0) uniform sampler2D u_input_texture;
uniform float strength;
vec2 texel_size = 1.0 / textureSize(u_input_texture, 0);

layout(location = 0) out vec3 fragColor;

void main() {
    vec3 result = vec3(0.0);

    result += texture(u_input_texture, uv + texel_size * vec2(-1, -1)).rgb * 1.0;
    result += texture(u_input_texture, uv + texel_size * vec2(0, -1)).rgb * 2.0;
    result += texture(u_input_texture, uv + texel_size * vec2(1, -1)).rgb * 1.0;
    result += texture(u_input_texture, uv + texel_size * vec2(-1, 0)).rgb * 2.0;
    result += texture(u_input_texture, uv).rgb * 4.0;
    result += texture(u_input_texture, uv + texel_size * vec2(1, 0)).rgb * 2.0;
    result += texture(u_input_texture, uv + texel_size * vec2(-1, 1)).rgb * 1.0;
    result += texture(u_input_texture, uv + texel_size * vec2(0, 1)).rgb * 2.0;
    result += texture(u_input_texture, uv + texel_size * vec2(1, 1)).rgb * 1.0;

    result /= 16.0;

    fragColor = strength * result;
}
#endif
