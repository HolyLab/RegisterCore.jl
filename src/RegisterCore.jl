module RegisterCore

using CenterIndexedArrays: CenterIndexedArrays, CenterIndexedArray
using ImageCore: ImageCore, Gray, gray
using ImageFiltering: ImageFiltering, KernelFactors, NA, imfilter
using Requires: Requires, @require

import Base: +, -, *, /

export
    # types
    MismatchArray,
    NumDenom,
    ColonFun,
    PreprocessSNF,
    # functions
    argmin_mismatch,
    highpass,
    highpass!,
    maxshift,
    mismatcharrays,
    ratio,
    separate,
    paddedview,
    trimmedview

"""
Core types and utilities for image registration mismatch computations.

**Types**
- [`NumDenom`](@ref): a packed `(numerator, denominator)` pair; supports vector-space
  arithmetic for use with `Interpolations.jl`
- [`MismatchArray`](@ref): a `CenterIndexedArray` of `NumDenom` elements representing
  mismatch over a range of shifts
- [`PreprocessSNF`](@ref): shot-noise–filtering image preprocessor (bias subtract,
  square-root transform, band-pass filter)
- [`ColonFun`](@ref): callable singleton returning `Colon()`, for type-stable slicing

**Functions**
- [`separate`](@ref): unpack a `NumDenom` or `MismatchArray` into `(num, denom)` arrays
- [`ratio`](@ref): convert `NumDenom` to a scalar ratio, with threshold masking
- [`maxshift`](@ref): return the maximum-shift half-size of a `MismatchArray`
- [`mismatcharrays`](@ref): pack array-of-arrays pairs into an array of `MismatchArray`s
- [`argmin_mismatch`](@ref): find the shift index of the minimum mismatch
- [`highpass`](@ref) / [`highpass!`](@ref): high-pass filter an image (Gaussian-based, NaN-safe)
- [`paddedview`](@ref) / [`trimmedview`](@ref): extend/trim a `SubArray` to/from its parent
"""
RegisterCore

"""
    x = NumDenom(num, denom)

A packed `(num, denom)` pair with element type `T`. Access fields as `x.num` and
`x.denom`. Arguments of type `Gray` are automatically unwrapped; mismatched numeric
types are promoted.

`NumDenom` objects act as 2-vectors under arithmetic — addition and scalar
multiplication operate component-wise on both fields:

    nd1 + nd2  →  NumDenom(nd1.num + nd2.num, nd1.denom + nd2.denom)
    nd1 - nd2  →  NumDenom(nd1.num - nd2.num, nd1.denom - nd2.denom)
    c * nd     →  NumDenom(c*nd.num, c*nd.denom)
    nd / c     →  NumDenom(nd.num/c, nd.denom/c)

This vector-space algebra (rather than ratio algebra) is intentional: it allows
`Interpolations.jl` to interpolate numerator and denominator jointly without
recomputing interpolation weights, enabling efficient aperture-windowed mismatch
computations.

As a consequence, `convert(Float64, ::NumDenom)` is deliberately not defined, because
the component-wise arithmetic breaks the ratio interpretation. Use [`ratio`](@ref) to
convert to a scalar ratio.
"""
struct NumDenom{T <: Number}
    num::T
    denom::T
end
NumDenom(n::Gray, d::Gray) = NumDenom(gray(n), gray(d))
NumDenom(n::Gray, d) = NumDenom(gray(n), d)
NumDenom(n, d::Gray) = NumDenom(n, gray(d))
NumDenom(n, d) = NumDenom(promote(n, d)...)

(+)(p1::NumDenom, p2::NumDenom) = NumDenom(p1.num + p2.num, p1.denom + p2.denom)
(-)(p1::NumDenom, p2::NumDenom) = NumDenom(p1.num - p2.num, p1.denom - p2.denom)
(*)(n::Number, p::NumDenom) = NumDenom(n * p.num, n * p.denom)
(*)(p::NumDenom, n::Number) = n * p
(/)(p::NumDenom, n::Number) = NumDenom(p.num / n, p.denom / n)
Base.oneunit(::Type{NumDenom{T}}) where {T} = NumDenom{T}(oneunit(T), oneunit(T))
Base.oneunit(p::NumDenom) = oneunit(typeof(p))
Base.zero(::Type{NumDenom{T}}) where {T} = NumDenom(zero(T), zero(T))
Base.zero(p::NumDenom) = zero(typeof(p))
Base.promote_rule(::Type{NumDenom{T1}}, ::Type{T2}) where {T1, T2 <: Number} = NumDenom{promote_type(T1, T2)}
Base.promote_rule(::Type{NumDenom{T1}}, ::Type{NumDenom{T2}}) where {T1, T2} = NumDenom{promote_type(T1, T2)}
Base.eltype(::Type{NumDenom{T}}) where {T} = T
Base.convert(::Type{NumDenom{T}}, p::NumDenom{T}) where {T} = p
Base.convert(::Type{NumDenom{T}}, p::NumDenom) where {T} = NumDenom{T}(p.num, p.denom)

Base.round(::Type{NumDenom{T}}, p::NumDenom) where {T} = NumDenom{T}(round(T, p.num), round(T, p.denom))

Base.convert(::Type{F}, p::NumDenom) where {F <: AbstractFloat} = error("`convert($F, ::NumDenom)` is deliberately not defined, see `?NumDenom`.")

function Base.show(io::IO, p::NumDenom)
    print(io, "NumDenom(")
    show(io, p.num)
    print(io, ",")
    show(io, p.denom)
    print(io, ")")
    return
end

"""
    MismatchArray{ND,N,A}

A `CenterIndexedArray` whose elements are [`NumDenom`](@ref) pairs, representing
packed numerator/denominator mismatch data over a range of shifts.

The axes are centered at zero, so `axes(D, k)` runs from `-maxshift(D)[k]` to
`+maxshift(D)[k]`; an index value of `(i, j, ...)` corresponds to a shift of
`(i, j, ...)` pixels.

    D = MismatchArray(num::AbstractArray, denom::AbstractArray)

Pack same-size arrays `num` and `denom` into a `MismatchArray` with centered axes of
half-size `size(num) .÷ 2`. Element type is `NumDenom{promote_type(eltype(num), eltype(denom))}`.

    D = MismatchArray(T, dims::Dims)
    D = MismatchArray(T, dims::Integer...)

Allocate an uninitialized `MismatchArray` with element type `NumDenom{T}` and the given
dimensions. Useful for pre-allocating output before filling with `copyto!`.
"""
const MismatchArray{ND <: NumDenom, N, A} = CenterIndexedArray{ND, N, A}

"""
    mxs = maxshift(D::MismatchArray)

Return the half-size of the `MismatchArray` `D` as a tuple of integers — i.e., the
maximum shift (in pixels, per dimension) that was searched when computing the mismatch.
Equivalent to `D.halfsize`.
"""
maxshift(A::MismatchArray) = A.halfsize

function (::Type{M})(num::AbstractArray, denom::AbstractArray) where {M <: MismatchArray}
    size(num) == size(denom) || throw(DimensionMismatch("num and denom must have the same size"))
    T = promote_type(eltype(num), eltype(denom))
    numdenom = CenterIndexedArray{NumDenom{T}}(undef, size(num))
    return _packnd!(numdenom, num, denom)
end

function _packnd!(numdenom::AbstractArray, num::AbstractArray, denom::AbstractArray)
    Rnd, Rnum, Rdenom = eachindex(numdenom), eachindex(num), eachindex(denom)
    if Rnum == Rdenom
        for (Idest, Isrc) in zip(Rnd, Rnum)
            @inbounds numdenom[Idest] = NumDenom(num[Isrc], denom[Isrc])
        end
    elseif Rnd == Rnum
        for (Inum, Idenom) in zip(Rnum, Rdenom)
            @inbounds numdenom[Inum] = NumDenom(num[Inum], denom[Idenom])
        end
    else
        for (Ind, Inum, Idenom) in zip(Rnd, Rnum, Rdenom)
            @inbounds numdenom[Ind] = NumDenom(num[Inum], denom[Idenom])
        end
    end
    return numdenom
end

function _packnd!(numdenom::CenterIndexedArray, num::CenterIndexedArray, denom::CenterIndexedArray)
    @simd for I in eachindex(num)
        @inbounds numdenom[I] = NumDenom(num[I], denom[I])
    end
    return numdenom
end

# The next are mostly used just for testing
"""
    mms = mismatcharrays(nums, denom::AbstractArray{<:Number})
    mms = mismatcharrays(nums, denoms::AbstractArray{<:AbstractArray})

Pack array-of-arrays `(num, denom)` pairs into an `Array{MismatchArray}`.

The first form uses the same `denom` array for every entry in `nums`. The second form
pairs each `nums[i]` with the corresponding `denoms[i]`; `nums` and `denoms` must have
the same size.

Returns an `Array{MismatchArray}` with the same shape as `nums`.
"""
function mismatcharrays(nums::AbstractArray{A}, denom::AbstractArray{T}) where {A <: AbstractArray, T <: Number}
    first = true
    local mms
    for i in eachindex(nums)
        num = nums[i]
        mm = MismatchArray(num, denom)
        if first
            mms = Array{typeof(mm)}(undef, size(nums))
            first = false
        end
        mms[i] = mm
    end
    return mms
end

function mismatcharrays(nums::AbstractArray{A1}, denoms::AbstractArray{A2}) where {A1 <: AbstractArray, A2 <: AbstractArray}
    size(nums) == size(denoms) || throw(DimensionMismatch("nums and denoms arrays must have the same number of apertures"))
    first = true
    local mms
    for i in eachindex(nums, denoms)
        mm = MismatchArray(nums[i], denoms[i])
        if first
            mms = Array{typeof(mm)}(undef, size(nums))
            first = false
        end
        mms[i] = mm
    end
    return mms
end

"""
    num, denom = separate(data::AbstractArray{NumDenom{T}})
    num, denom = separate(mm::MismatchArray)
    nums, denoms = separate(mma::AbstractArray{<:MismatchArray})

Split packed [`NumDenom`](@ref) data into separate numerator and denominator arrays.

- For a plain `AbstractArray{NumDenom{T}}`, returns a pair of `Array{T}` with the same
  size and linear indexing as `data`.
- For a [`MismatchArray`](@ref), returns a pair of `CenterIndexedArray{T}`, preserving
  the centered axes (so index ranges correspond to shift values).
- For an array of `MismatchArray`s, returns a pair of `Array{CenterIndexedArray{T}}`.
"""
function separate(data::AbstractArray{NumDenom{T}}) where {T}
    num = Array{T}(undef, size(data))
    denom = similar(num)
    for I in eachindex(data)
        nd = data[I]
        num[I] = nd.num
        denom[I] = nd.denom
    end
    return num, denom
end

function separate(mm::MismatchArray)
    num, denom = separate(mm.data)
    return CenterIndexedArray(num), CenterIndexedArray(denom)
end

function separate(mma::AbstractArray{M}) where {M <: MismatchArray}
    T = eltype(eltype(M))
    nums = Array{CenterIndexedArray{T, ndims(M)}}(undef, size(mma))
    denoms = similar(nums)
    for (i, mm) in enumerate(mma)
        nums[i], denoms[i] = separate(mm)
    end
    return nums, denoms
end

"""
    r = ratio(nd::NumDenom, thresh; fillval=NaN)
    r = ratio(r::Real, thresh; fillval=NaN)

Return the ratio `nd.num/nd.denom`, or `fillval` (converted to the ratio type) when
`nd.denom < thresh`. Setting `thresh = 0` always returns the ratio.

The second form accepts a plain `Real` and returns it unchanged, regardless of `thresh`
and `fillval` — `thresh` and `fillval` are silently ignored. This allows callers to
handle both [`NumDenom`](@ref) and pre-computed ratio arrays uniformly without
branching on the element type.
"""
@inline function ratio(nd::NumDenom{T}, thresh; fillval = convert(T, NaN)) where {T}
    r = nd.num / nd.denom
    return nd.denom < thresh ? oftype(r, fillval) : r
end
ratio(r::Real, thresh; fillval = NaN) = r

@deprecate(ratio(nd::NumDenom{T}, thresh, fillval) where {T}, ratio(nd, thresh; fillval=fillval))
@deprecate(ratio(r::Real, thresh, fillval), ratio(r, thresh))

(::Type{M})(::Type{T}, dims::Dims) where {M <: MismatchArray, T} = CenterIndexedArray{NumDenom{T}}(undef, dims)
(::Type{M})(::Type{T}, dims::Integer...) where {M <: MismatchArray, T} = CenterIndexedArray{NumDenom{T}}(undef, dims)

function Base.copyto!(M::MismatchArray, nd::Tuple{AbstractArray, AbstractArray})
    num, denom = nd
    size(M) == size(num) == size(denom) || error("all sizes must match")
    for (IM, Ind) in zip(eachindex(M), eachindex(num))
        M[IM] = NumDenom(num[Ind], denom[Ind])
    end
    return M
end


#### Utility functions ####

"""
    index = argmin_mismatch(numdenom::MismatchArray, thresh::Real)
    index = argmin_mismatch(r::CenterIndexedArray{<:Number})

Return the `CartesianIndex` of the minimum mismatch, excluding edge points.

The first form operates on a [`MismatchArray`](@ref): it computes `num/denom` at each
interior point and finds the minimum among points where `denom > thresh`. The second
form operates on a pre-computed ratio array `r` of plain numbers and finds the interior
minimum unconditionally.

"Edge points" are the outermost index value on each side of every dimension; they are
always excluded from consideration.

If no valid point is found (e.g., all `denom ≤ thresh`), returns a zero
`CartesianIndex` — check for this case before using the result as an array index.
"""
function argmin_mismatch(numdenom::MismatchArray{NumDenom{T}, N}, thresh::Real) where {T, N}
    imin = CartesianIndex(ntuple(d -> 0, Val(N)))
    rmin = typemax(T)
    threshT = convert(T, thresh)
    @inbounds for I in CartesianIndices(map(trimedges, axes(numdenom)))
        nd = numdenom[I]
        if nd.denom > threshT
            r = nd.num / nd.denom
            if r < rmin
                imin = I
                rmin = r
            end
        end
    end
    return imin
end

function argmin_mismatch(r::CenterIndexedArray{T, N}) where {T <: Number, N}
    imin = CartesianIndex(ntuple(d -> 0, Val(N)))
    rmin = typemax(T)
    @inbounds for I in CartesianIndices(map(trimedges, axes(r)))
        rval = r[I]
        if rval < rmin
            imin = I
            rmin = rval
        end
    end
    return imin
end

trimedges(r::AbstractUnitRange) = (first(r) + 1):(last(r) - 1)

### Miscellaneous

"""
    datahp = highpass([T], data, sigma)
    highpass!(out, data, sigma)
    highpass!(data, sigma)

Return (or write in-place) a high-pass–filtered version of `data` with negative values
clamped to zero. The high-pass is computed by subtracting a Gaussian-smoothed copy of
`data` (via `ImageFiltering.jl`), which gracefully handles `NaN` values.

`sigma` must be an iterable (e.g., a tuple or vector) with one width per dimension of
`data`. To skip filtering along a particular axis, set the corresponding entry to `Inf`
(the input is then returned as-is, converted to `Array{T}`, with no subtraction or
clamping applied).

For `highpass`, the optional first argument `T` sets the element type of the output:
- `highpass(T, data, sigma)` — allocates an output of element type `T`
- `highpass(data, sigma)` — `T` defaults to `eltype(data)` for `AbstractFloat` inputs,
  or `Float32` for `Integer`/`FixedPoint` inputs

For `highpass!`, the output element type is `eltype(out)`:
- `highpass!(out, data, sigma)` — writes result into `out`; `out` and `data` may be
  distinct arrays (useful for pre-allocated buffers in hot loops)
- `highpass!(data, sigma)` — filters `data` in-place

See also [`PreprocessSNF`](@ref) for a combined shot-noise–filtering preprocessor.
"""
function highpass(::Type{T}, data::AbstractArray, sigma) where {T}
    if any(isinf, sigma)
        datahp = convert(Array{T, ndims(data)}, data)
    else
        datahp = data - imfilter(T, data, KernelFactors.IIRGaussian(T, (sigma...,)), NA())
    end
    datahp[datahp .< 0] .= 0  # truncate anything below 0
    return datahp
end
highpass(data::AbstractArray{T}, sigma) where {T <: AbstractFloat} = highpass(T, data, sigma)
highpass(data::AbstractArray, sigma) = highpass(Float32, data, sigma)

"""
    highpass!(out, data, sigma)
    highpass!(data, sigma)

In-place variant of [`highpass`](@ref). See that function for full documentation.
"""
function highpass!(out::AbstractArray, data::AbstractArray, sigma)
    T = eltype(out)
    if any(isinf, sigma)
        out .= data
    else
        out .= data .- imfilter(T, data, KernelFactors.IIRGaussian(T, (sigma...,)), NA())
    end
    out[out .< 0] .= 0
    return out
end
highpass!(data::AbstractArray, sigma) = highpass!(data, data, sigma)

"""
    pp = PreprocessSNF(bias, sigmalp, sigmahp)

Construct a shot-noise–filtered preprocessor. Call it as `pp(img)` to preprocess an
image. All arguments are stored as `Float32` (or `Vector{Float32}`), so `Float64`
inputs are silently converted.

The "SNF" name stands for "shot-noise filtered": this preprocessor is designed for
photon-counting (Poisson) data, where the variance equals the mean.

Processing steps:
1. Subtract `bias` and clamp to zero: `A′ = max(0, img - bias)`
2. Square-root transform to stabilize variance: `A″ = √A′`
3. High-pass filter with Gaussian width `sigmahp` (subtracts a low-pass Gaussian)
4. Low-pass filter with Gaussian width `sigmalp`

`sigmalp` and `sigmahp` must each be a vector of length `ndims(img)`. Pass
`sigmalp = zeros(ndims(img))` to skip low-pass filtering; pass
`sigmahp = fill(Inf, ndims(img))` to skip high-pass filtering.

When called on a `SubArray`, `pp` extends the view to the full parent (via
[`paddedview`](@ref)) before processing, then trims the result back to the original
index range (via [`trimmedview`](@ref)). This preserves any padding context around
the sub-region.

If `ImageMetadata` is loaded, `pp` also accepts `ImageMeta` arrays and propagates
image properties to the output.
"""
mutable struct PreprocessSNF  # Shot-noise filtered
    const bias::Float32
    const sigmalp::Vector{Float32}
    const sigmahp::Vector{Float32}
end
# PreprocessSNF(bias::T, sigmalp, sigmahp) = PreprocessSNF{T}(bias, T[sigmalp...], T[sigmahp...])

function preprocess(pp::PreprocessSNF, A::AbstractArray)
    Af = sqrt_subtract_bias(A, pp.bias)
    return imfilter(highpass(Af, pp.sigmahp), KernelFactors.IIRGaussian((pp.sigmalp...,)), NA())
end
(pp::PreprocessSNF)(A::AbstractArray) = preprocess(pp, A)
# For SubArrays, extend to the parent along any non-sliced
# dimension. That way, we keep any information from padding.
function (pp::PreprocessSNF)(A::SubArray)
    Bpad = preprocess(pp, paddedview(A))
    return trimmedview(Bpad, A)
end
# ImageMeta method is defined under @require in __init__

function sqrt_subtract_bias(A, bias)
    #    T = typeof(sqrt(one(promote_type(eltype(A), typeof(bias)))))
    T = Float32
    out = Array{T}(undef, size(A))
    for I in eachindex(A)
        @inbounds out[I] = sqrt(max(zero(T), convert(T, A[I]) - bias))
    end
    return out
end


"""
    Apad = paddedview(A::SubArray)

Return a `SubArray` of `A`'s parent that extends to the full parent size along every
dimension of `A` that was indexed by a `UnitRange`. Dimensions indexed by a scalar are
still dropped (as usual for `SubArray`), and dimensions indexed by a `Slice` are
unchanged.

Only `Slice`, `Real`, and `UnitRange` index types are supported; any other index type
throws an error.

See also [`trimmedview`](@ref).
"""
paddedview(A::SubArray) = _paddedview(A, (), (), A.indices...)
_paddedview(A::SubArray{T, N, P, I}, newindexes, newsize) where {T, N, P, I} =
    SubArray(A.parent, newindexes)
@inline function _paddedview(A, newindexes, newsize, index, indexes...)
    d = length(newindexes) + 1
    return _paddedview(A, (newindexes..., pdindex(A.parent, d, index)), pdsize(A.parent, newsize, d, index), indexes...)
end
pdindex(A, d, i::Base.Slice) = i
pdindex(A, d, i::Real) = i
pdindex(A, d, i::UnitRange) = 1:size(A, d)
pdindex(A, d, i) = error("Cannot pad with an index of type ", typeof(i))

pdsize(A, newsize, d, i::Base.Slice) = tuple(newsize..., size(A, d))
pdsize(A, newsize, d, i::Real) = newsize
pdsize(A, newsize, d, i::UnitRange) = tuple(newsize..., size(A, d))

"""
    B = trimmedview(Bpad::AbstractArray, A::SubArray)

Return a view `B` of `Bpad` with `axes(B) == axes(A)`.

`Bpad` must be an `AbstractArray` with the same size as `paddedview(A)` — typically
it is the result of an operation applied to `paddedview(A)`. Dimensions of `A` that
were indexed by a scalar are skipped (they are dropped in `A`); all other dimensions
are sliced with the original index range from `A`.

See also [`paddedview`](@ref).
"""
function trimmedview(Bpad::AbstractArray, A::SubArray)
    ndims(Bpad) == ndims(A) || throw(DimensionMismatch("dimensions $(ndims(Bpad)) and $(ndims(A)) of Bpad and A must match"))
    return _trimmedview(Bpad, A.parent, 1, (), A.indices...)
end
_trimmedview(Bpad, P, d, newindexes) = view(Bpad, newindexes...)
@inline _trimmedview(Bpad, P, d, newindexes, index::Real, indexes...) =
    _trimmedview(Bpad, P, d + 1, newindexes, indexes...)
@inline function _trimmedview(Bpad, P, d, newindexes, index, indexes...)
    dB = length(newindexes) + 1
    Bsz = size(Bpad, dB)
    Psz = size(P, d)
    Bsz == Psz || throw(DimensionMismatch("dimension $dB of Bpad has size $Bsz, should have size $Psz"))
    return _trimmedview(Bpad, P, d + 1, (newindexes..., index), indexes...)
end

"""
    ColonFun()(i::Int) -> Colon()

A callable singleton that returns `Colon()` for any integer argument.
Used in place of `Colon()` directly where type-stable, composable slicing
is needed (e.g., constructing index tuples programmatically).
"""
struct ColonFun end
ColonFun(::Int) = Colon()

function __init__()
    return @require ImageMetadata = "bc367c6b-8a6b-528e-b4bd-a4b897500b49" begin
        (pp::PreprocessSNF)(A::ImageMetadata.ImageMeta) = ImageMetadata.shareproperties(A, pp(ImageMetadata.arraydata(A)))
    end
end

end
