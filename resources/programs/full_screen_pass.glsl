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

layout(location = 0) out vec3 fragColor;

void main() {
    fragColor = texture(u_input_texture, uv).rgb;
}
#endif
