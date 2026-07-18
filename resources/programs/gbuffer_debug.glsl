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

out vec4 fragColor;
uniform sampler2D u_albedo;
uniform sampler2D u_normal;
uniform sampler2D u_depth;
uniform sampler2D u_motion_vectors;
uniform sampler2D u_lighting;
uniform bool full;
const float exposure = 1.0;
const float gamma = 2.4;

in vec2 uv;

vec3 tonemap(vec3 hdr) {
    hdr *= exposure;
    vec3 ldr = hdr / (hdr + vec3(1.0));
    return ldr;
}

vec3 lumaBasedReinhardToneMapping(vec3 color)
{
    float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float toneMappedLuma = luma / (1. + luma);
    color *= toneMappedLuma / luma;
    color = pow(color, vec3(1. / gamma));
    return color;
}

void main() {
    vec2 local = fract(uv * 2.0);
    vec3 color;

    if (uv.x < 0.5)
    {
        if (uv.y < 0.5)
        {
            float depth = texture(u_depth, local).r / 1000.0;
            color = vec3(depth);
        }
        else
        {
            vec3 motion = abs(texture(u_motion_vectors, local).rgb) * 10.0;
            vec3 albedo = texture(u_albedo, local).rgb;
            color = mix(albedo, motion, 0.00001);
        }
    }
    else
    {
        if (uv.y < 0.5)
        {
            float checker = mod(floor(gl_FragCoord.x / 32.0) + floor(gl_FragCoord.y / 32.0), 2.0);
            color = mix(vec3(0.1), vec3(0.9), checker);
            color = lumaBasedReinhardToneMapping(texture(u_lighting, local).rgb);
        }
        else
        {
            vec3 normal = texture(u_normal, local).rgb;
            color = normal * 0.5 + 0.5;
        }
    }
    if (full) {
        fragColor = vec4(texture(u_lighting, uv).rgb, 1.0);
        fragColor = vec4(tonemap(texture(u_lighting, uv).rgb), 1.0);
    } else {
        fragColor = vec4(color, 1.0);
    }
}
#endif
