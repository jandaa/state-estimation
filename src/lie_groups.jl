import LinearAlgebra

function close_to_zero(value::T) where {T<:AbstractFloat}
    return abs(value) < 1e-2
end

function se3_inv(T::Matrix{<:AbstractFloat})
    C = T[1:3, 1:3]
    d = T[1:3, 4]
    return [C' -C'*d; 0 0 0 1]
end

# function skew(x::Vector{<:AbstractFloat})
#     return [
#         0 -x[3] x[2]
#         x[3] 0 -x[1]
#         -x[2] x[1] 0
#     ]
# end

function so3_wedge(ϕ::Vector{<:AbstractFloat})
    return [
        0 -ϕ[3] ϕ[2]
        ϕ[3] 0 -ϕ[1]
        -ϕ[2] ϕ[1] 0
    ]
end

function so3_exp(psi::Vector{<:AbstractFloat})
    angle = norm(psi, 2)
    axis = psi / angle
    s = sin(angle)
    c = cos(angle)
    return c * I(3) + (1 - c) * axis .* axis' - s * so3_wedge(axis)
end

function so3_vee(C::Matrix{<:AbstractFloat})
    return [C[3, 2], C[1, 3], C[2, 1]]
end

function so3_log(C::Matrix{<:AbstractFloat})

    cos_angle = 0.5 * tr(C) - 0.5
    cos_angle = clamp(cos_angle, -1.0, 1.0)
    angle = acos(cos_angle)

    if close_to_zero(angle)
        tmp = C - I
    else
        tmp = (angle / (2 * sin(angle))) * (C - C')
    end

    return so3_vee(tmp)
end

function so3_jacobian(psi::Vector{<:AbstractFloat})
    angle = norm(psi)

    if close_to_zero(angle)
        return I + 0.5 * so3_wedge(psi)
    end

    axis = psi / angle
    s = sin(angle)
    c = cos(angle)

    return (s / angle) * I(3) + (1 - s / angle) * axis * axis' + (1 - c) / angle * so3_wedge(axis)
end

function so3_inv_left_jacobian(phi::Vector{<:AbstractFloat})
    dof = 3
    angle = norm(phi)

    # Near phi==0, use first order Taylor expansion
    if close_to_zero(angle)
        return I(3) - 0.5 * so3_wedge(phi)
    end

    axis = phi / angle
    half_angle = 0.5 * angle
    cot_half_angle = 1.0 / tan(half_angle)

    return half_angle * cot_half_angle * I(dof) + (1 - half_angle * cot_half_angle) * (axis * axis') - half_angle * so3_wedge(axis)

end

# Might have to update this
function se3_exp(d::Vector{<:AbstractFloat}, psi::Vector{<:AbstractFloat})
    J = so3_jacobian(psi)
    C = so3_exp(psi)
    return [C J*d; 0 0 0 1]
end

function se3_log(T::Matrix{<:AbstractFloat})
    psi = so3_log(T[1:3, 1:3])

    J = so3_inv_left_jacobian(psi)
    r = inv(J) * T[1:3, 4]

    return [r; psi]
end

# Functions taken from the python liegroups packages

function se3_wedge(ξ::Vector{<:AbstractFloat})
    dim = 4
    Ξ = zeors(size(ξ, 1), dim, dim)
    Ξ[:, 1:3, 1:3] = so3_wedge(ξ[:, 4:6])
    Ξ[:, 0:3, 3] = ξ[:, 1:3]
    return Ξ
end

function se3_odot(p::Vector{<:AbstractFloat})
    dim = 4
    dof = 6

    result = zeros(size(p, 1), dof)

    if size(p, 1) == dim - 1
        result[1:3, 1:3] = I(3)
        result[1:3, 4:6] = so3_wedge(-p)
    else
        result[1:3, 1:3] = I(3)
        result[1:3, 4:6] = so3_wedge(-p)
    end

    return result
end

function se3_adjoint(T::Matrix{<:AbstractFloat})
    C = T[1:3, 1:3]
    r = T[1:3, 4]
    return [
        C so3_wedge(r)*C
        zeros(3, 3) C
    ]
end