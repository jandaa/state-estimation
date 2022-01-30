import LinearAlgebra

function skew(x::Vector{<:AbstractFloat})
    return [
        0 -x[3] x[2]
        x[3] 0 -x[1]
        -x[2] x[1] 0
    ]
end

function so3_exp(psi::Vector{<:AbstractFloat})
    angle = norm(psi, 2)
    axis = psi / angle
    s = sin(angle)
    c = cos(angle)
    return c * I(3) + (1 - c) * axis .* axis' - s * skew(axis)
end

function so3_log(C::Matrix{<:AbstractFloat})

    cos_angle = 0.5 * tr(C) - 0.5
    cos_angle = clamp(cos_angle, -1.0, 1.0)
    angle = acos(cos_angle)

    if abs(angle) < 0.001
        tmp = C - I
    else
        tmp = (angle / (2 * sin(angle))) * (C - C')
    end

    return [tmp[3, 2], tmp[1, 3], tmp[2, 1]]
end

function so3_jacobian(psi::Vector{<:AbstractFloat})
    angle = norm(psi)

    if abs(angle) < 0.001
        return I + 0.5 * skew(psi)
    end

    axis = psi / angle
    s = sin(angle)
    c = cos(angle)

    return (s / angle) * I(3) + (1 - s / angle) * axis * axis' + (1 - c) / angle * skew(axis)
end

function se3_exp(d::Vector{<:AbstractFloat}, psi::Vector{<:AbstractFloat})
    J = so3_jacobian(psi)
    C = so3_exp(psi)

    t = J * d

    return [C d; 0 0 0 1]
end

function se3_log(T::Matrix{<:AbstractFloat})
    psi = so3_log(T[1:3, 1:3])

    J = so3_jacobian(psi)
    r = inv(J) * T[1:3, 4]

    return [r; psi]
end