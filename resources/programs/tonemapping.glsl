#version 330

#if defined VERTEX_SHADER

in vec3 in_position;
in vec2 in_texcoord_0;

out vec2 uv;

void main() {
    gl_Position = vec4(in_position, 1);
    uv = in_texcoord_0;
}

#elif defined FRAGMENT_SHADER

in vec2 uv;

uniform sampler2D u_final_texture;
uniform float gamma = 0.8;

out vec4 fragColor;

vec3 lumaBasedReinhardToneMapping(vec3 color)
{
    float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float toneMappedLuma = luma / (1. + luma);
    color *= toneMappedLuma / luma;
    color = pow(color, vec3(1. / gamma));
    return color;
}

void main() {
    fragColor = vec4(lumaBasedReinhardToneMapping(texture(u_final_texture, uv).rgb), 1.0);
}
#endif
