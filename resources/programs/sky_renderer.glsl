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

uniform mat4 uInvView;
uniform mat4 uInvProjection;

out vec4 fragColor;

in vec2 vUV;

vec3 skyColor(vec3 rayDir)
{
    vec3 dir = normalize(rayDir);

    // Map y from [-1,1] to [0,1]
    float t = 0.5 * (dir.y + 1.0);

    vec3 horizon = vec3(0.8, 0.9, 1.0);
    vec3 zenith  = vec3(0.2, 0.4, 0.8);

    return mix(horizon, zenith, t) * 2.5;
}

void main() {
    vec2 ndc = vUV * 2.0 - 1.0;
    vec4 clip = vec4(ndc, -1.0, 1.0);
    vec4 eye = uInvProjection * clip;
    eye = vec4(eye.xy, -1.0, 0.0);

    vec3 rd = normalize((uInvView * eye).xyz);
    fragColor = vec4(skyColor(rd), 1.0);
}
#endif
