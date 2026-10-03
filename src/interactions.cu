#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

#define RAY_NUDGE 0.001f

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ glm::vec3 calculateGlassDirection(
    glm::vec3 dir,
    glm::vec3 normal,
    bool comingFromOutside,
    float ior,
    thrust::default_random_engine &rng)
{
    // normal always points against the ray
    float eta = comingFromOutside ? 1.0f / ior : ior; // we're always assuming we're in air (ior = 1)
    float cosTheta = glm::min(-glm::dot(dir, normal), 1.0f);  // the dot will always be negative.

    // use's snell's law
    glm::vec3 refracted = glm::refract(dir, normal, eta);

    // total internal reflection. everything reflects
    if (glm::dot(refracted, refracted) == 0.0f)
    {
        return glm::reflect(dir, normal);
    }

    // normal case: shlick's approximation
    // get the angle on the air side, which is the refracted one when going out
    float c = comingFromOutside ? cosTheta : glm::dot(refracted, -normal);
    float R0 = (1.0f - ior) / (1.0f + ior);
    R0 *= R0;
    float R = R0 + (1.0f - R0) * powf(1.0f - c, 5.0f);

    // pick reflect with probability R, refract w prob 1-R.
    thrust::uniform_real_distribution<float> u01(0, 1);
    return u01(rng) < R ? glm::reflect(dir, normal) : refracted;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    bool comingFromOutside,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    glm::vec3 newDir;

    // glass: reflect with probability R, otherwise refract
    if (m.hasRefractive > 0.0f)
    {
        newDir = calculateGlassDirection(pathSegment.ray.direction, normal, comingFromOutside, m.indexOfRefraction, rng);
    }

    // perfect mirror
    else if (m.hasReflective > 0.0f)
    {
        newDir = glm::reflect(pathSegment.ray.direction, normal);
    }
    
    // diffuse. we're just sampling a cosine-weighted random direction around the normal, this is the monte carlo estimation
    else
    {
        newDir = calculateRandomDirectionInHemisphere(normal, rng);
    }

    // for all three paths, the color update simplifies to this 
    pathSegment.color *= m.color;

    // avoid floating point issues/shadow acne with a nudge
    // which way to nudge? all (non refractive) cases will reflect off/point in the same direction as the normal, so nudge in that direction
    // in the refractive cases, its the opposite so nudge in
    float side = glm::dot(newDir, normal) > 0.0f ? 1.0f : -1.0f;
    pathSegment.ray.origin = intersect + normal * side * RAY_NUDGE;
    pathSegment.ray.direction = glm::normalize(newDir);
}
