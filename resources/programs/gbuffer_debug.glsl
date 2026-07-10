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
uniform sampler2D u_lighting;
const float exposure = 1.0;

in vec2 uv;

vec3 decodeNormalRGB10A2(vec3 encoded)
{
    // map [0,1] -> [-1,1]
    return normalize(encoded * 2.0 - 1.0);
}

void main() {
    vec2 local = fract(uv * 2.0);
    vec3 color;

    if (uv.x < 0.5)
    {
        if (uv.y < 0.5)
        {
            float depth = texture(u_depth, local).r;
            color = vec3(depth);
        }
        else
        {
            color = texture(u_albedo, local).rgb;
        }
    }
    else
    {
        if (uv.y < 0.5)
        {
            float checker = mod(floor(gl_FragCoord.x / 32.0) + floor(gl_FragCoord.y / 32.0), 2.0);
            color = mix(vec3(0.1), vec3(0.9), checker);
            color = texture(u_lighting, local).rgb;
        }
        else
        {
            vec3 normal = decodeNormalRGB10A2(texture(u_normal, local).rgb);
            color = normal * 0.5 + 0.5;
        }
    }
    fragColor = vec4(color, 1.0);
}
#endif
