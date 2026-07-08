#version 330

#if defined VERTEX_SHADER

in vec3 in_position;
in vec2 in_texcoord_0;

out vec2 vUV;

void main() {
    gl_Position = vec4(in_position, 1);
    vUV = in_texcoord_0;
}

#elif defined FRAGMENT_SHADER

out vec4 fragColor;
uniform sampler2D u_texture;
const float exposure = 1.0;

in vec2 vUV;

void main() {
    vec3 hdr = texture(u_texture, vUV).rgb;
    hdr *= exposure;
    vec3 ldr = hdr / (hdr + vec3(1.0));
    fragColor = vec4(ldr, 1.0);
}
#endif
