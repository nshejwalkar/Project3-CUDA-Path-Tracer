#include "intersections.h"

#include <cfloat>

__host__ __device__ float boxIntersectionTest(
    const Geom& box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    const Geom& sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    // inverse transform the ray into object space
    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float triangleIntersectionTest(
    const Triangle& tri,
    Ray r,
    float& u,
    float& v)
{
    // everything happens in object space

    // glm's implementation of Möller-Trumbore is one-sided (false if a < epsilon). 
    // either we can copy the algo and edit, or (for now) just keep the meshes with a diffuse color so we don't need to worry about this
    glm::vec3 bary;
    float t;
    if (!glm::intersectRayTriangle(r.origin, r.direction, tri.v0, tri.v1, tri.v2, bary))  // calculates the face normal itself, does not know about our stored vertex normals
    {
        return -1;
    }

    // intersectRayTriangle returns barycentric coordinates, t 
    u = bary.x;
    v = bary.y;
    t = bary.z > 0.0f ? bary.z : -1;  // packed in
    return t;
}

__host__ __device__ bool aabbIntersectionTest(
    glm::vec3 bboxMin,
    glm::vec3 bboxMax,
    Ray r)
{
    // everything happens in object space

    // slab test, same idea as boxIntersectionTest: does the ray pass through this bbox?
    // inside the box = inside all 3 slabs at once, after the last entry and before the first exit
    float tNear = -FLT_MAX;
    float tFar = FLT_MAX;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        // for all x,y,z in parallel, find t_min and t_max s.t. ray.origin + t * ray.direction = (x,y,z)_(min,max).
        float t1 = (bboxMin[xyz] - r.origin[xyz]) / r.direction[xyz];
        float t2 = (bboxMax[xyz] - r.origin[xyz]) / r.direction[xyz];

        // same per axis, where the ray enters and leaves that slab
        tNear = glm::max(tNear, glm::min(t1, t2));
        tFar = glm::min(tFar, glm::max(t1, t2));

        // intervals already stopped overlapping, no need to check the other axes
        if (tNear > tFar)
        {
            return false;
        }
    }

    // does nay overlap between the three intervals exist? (and is that overlap ahead of the ray)
    return tFar > 0.0f;
}

__host__ __device__ float meshIntersectionTest(
    const Geom& mesh,
    Ray r,
    const Triangle* triangles,
    bool bboxCulling,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    // move ray into object space, same as box/sphere
    Ray q;
    q.origin    =                multiplyMV(mesh.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

    // ray doesn't even go inside the bbox
    if (bboxCulling && !aabbIntersectionTest(mesh.bboxMin, mesh.bboxMax, q))
    {
        return -1;
    }

    float tMin = FLT_MAX;
    float closestU = 0.0f;
    float closestV = 0.0f;
    int closestTri = -1;
    // we'll loop through all of the triangles for this mesh
    for (int i = mesh.triStart; i < mesh.triStart + mesh.triCount; i++)
    {
        float u, v;
        float t = triangleIntersectionTest(triangles[i], q, u, v);
        if (t > 0.0f && t < tMin)
        {
            tMin = t;
            closestU = u;
            closestV = v;
            closestTri = i;
        }
    }

    // ray does go inside the bbox but missed every triangle on the actual mesh
    if (closestTri == -1)
    {
        return -1;
    }

    // blend the vertex normals with the barycentric coordinates we received to get the smooth normal
    const Triangle& tri = triangles[closestTri];
    glm::vec3 objNormal = (1.0f - closestU - closestV) * tri.n0 + closestU * tri.n1 + closestV * tri.n2;

    // back to world space
    intersectionPoint = multiplyMV(mesh.transform, glm::vec4(getPointOnRay(q, tMin), 1.0f));
    normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(objNormal, 0.0f)));
    outside = true;  // triangleIntersectionTest only ever hits front faces

    return glm::length(r.origin - intersectionPoint);
}
