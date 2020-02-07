using RegisterCore
using CenterIndexedArrays, ImageCore, ImageMetadata, Interpolations
using Test

@testset "NumDenom and arrays" begin
    nd = NumDenom(3.5,10)
    @test ratio(nd, 5) == 3.5/10
    @test isnan(ratio(nd, 20))
    @test convert(NumDenom{Float32}, nd) === NumDenom(3.5f0,10)
    @test convert(typeof(nd), nd) === nd
    nd = NumDenom(3.5f0,10)
    @test isa(ratio(nd, 5), Float32)
    @test isa(ratio(nd, 20), Float32)
    col = Gray(0.2)
    @test NumDenom(col, 10).num === 0.2
    @test NumDenom(col, col).num === 0.2
    @test NumDenom(1, col).num === 1.0

    @test eltype(nd) === Float32
    @test zero(nd) === NumDenom{Float32}(0, 0)
    @test oneunit(nd) === NumDenom{Float32}(1, 1)

    io = IOBuffer()
    show(io, nd)
    @test String(take!(io)) == "NumDenom(3.5f0,10.0f0)"

    mm = MismatchArray(Float16, (3, 3))
    @test axes(mm) == Base.IdentityUnitRange.((-1:1, -1:1))

    numer, denom = rand(3,3), rand(3,3).+0.5
    mm = MismatchArray(numer, denom)
    r = CenterIndexedArray(numer./denom)
    @test ratio.(mm, 0.25) == r
    @test ratio.(r, 0.25) == r
    @test maxshift(mm) == (1, 1)

    mm = MismatchArray(CenterIndexedArray(numer), CenterIndexedArray(denom))
    @test ratio.(mm, 0.25) == r
    @test ratio(0.8, 1) === 0.8

    copyto!(mm, (numer, denom))
    @test ratio.(mm, 0.25) == r

    # The algebraic rules for interpolation (below) violate the notion that
    # a NumDenom is "equivalent" to the corresponding ratio.
    # Test that `convert` throws an error:
    @test_throws Exception convert(Float64, NumDenom(1, 2))

    # Interpolation of NumDenom arrays (and supporting or related arithmetic operations)
    @test 2.0*nd === nd*2.0 === NumDenom(7.0, 20.0)  # This multiplication rule is used in interpolation (also tests promotion)
    @test nd/2 === NumDenom(nd.num/2, nd.denom/2)
    @test nd - nd === NumDenom(0.0f0, 0.0f0)
    a = CenterIndexedArray([NumDenom(0, 0), NumDenom(1.0, 5.0), NumDenom(2.0, 2.0)])  # promotion
    @test eltype(a) === NumDenom{Float64}
    aitp = interpolate(a, BSpline(Linear()))
    @test ratio(aitp(-0.5), 1) ≈ 1/5
    @test ratio(aitp(+0.5), 1) ≈ 1.5/3.5

    @test round(NumDenom{Int}, NumDenom(1.2, 4.8)) === NumDenom{Int}(1, 5)

    # Finding the location of the minimum
    numer = [5,4,3,4.5,7].*[2,1,1.5,2,3]'
    numer[1,5] = -1  # on the edge, so it shouldn't be selected
    denom = ones(5,5)
    mma = MismatchArray(numer,denom)
    @test indmin_mismatch(mma, 0) == CartesianIndex((0,-1))
    denom = reshape(float(1:25), 5, 5)
    mma = MismatchArray(numer,denom)
    @test indmin_mismatch(mma, 0) == CartesianIndex((0,1))
    rat = ratio.(mma, 0.5)
    @test indmin_mismatch(rat) == CartesianIndex((0,1))
end

@testset "mismatcharrays and separate" begin
    nums = [rand(3,3), rand(3,3)]
    denom = ones(3,3)
    mms = mismatcharrays(nums, denom)
    @test mms[1] == MismatchArray(nums[1], denom)
    numss, denomss = separate(mms)
    @test numss == CenterIndexedArray.(nums)
    @test denomss == CenterIndexedArray.([denom, denom])
    @test separate(mms[1]) == (numss[1], denomss[1])

    mms2 = mismatcharrays(nums, [denom, denom])
    @test mms2 == mms
    raw = mms[1].data
    @test separate(raw) == (nums[1], denom)
end

@testset "highpass" begin
    a = [1, 1, 1, 1]
    ahp = highpass(a, (2,))
    @test all(x->abs(x)<1e-6, ahp)
    ahp = highpass(Float64, a, (2,))
    @test all(x->abs(x)<1e-12, ahp)

    af = float.(a)
    ahp = highpass(af, (2,))
    @test all(x->abs(x)<1e-12, ahp)
end

@testset "PreprocessSNF" begin
    A = [200 100; 10 1000]
    pp = PreprocessSNF(100, [0,0], [Inf,Inf])
    @test pp(A) ≈ [10 0; 0 30]

    B = fill(1000, 21, 17)
    B[10:11,8:9] .+= A
    pp = PreprocessSNF(100, [0,0], [3,3])
    ppB = pp(B)
    @test count(x->abs(x) < 1e-8, ppB) == length(B)-3
    @test count(x->abs(x) > 1, ppB[10:11,8:9]) == 3

    B = [1000*(isodd(i) ⊻ isodd(j)) for i = 1:21, j=1:17]
    pp = PreprocessSNF(100, [2,2], [Inf,Inf])
    ppB = pp(B)
    @test all(x->14<x<16, ppB)

    Bmeta = ImageMeta(B, date="today")
    @test isa(pp(Bmeta), ImageMeta)
end

@testset "Padding and trimming" begin
    # SubArray padding and trimming
    A = reshape(1:125, 5, 5, 5)
    S = view(A, 2:4, 1:3, 2)
    Spad = paddedview(S)
    @test Spad == A[:,:,2]
    S2 = trimmedview(A[:,:,2], S)
    @test S2 == S
    S = view(A, 2:4, 2, 1:3)
    Spad = paddedview(S)
    Aslice = view(A, :, 2, :)
    @test Spad == Aslice
    S2 = trimmedview(copy(Aslice), S)
    @test S2 == S
    S2 = trimmedview(Aslice, S)
    @test S2 == S

    S = view(A, :, 1:3, 2)
    Spad = paddedview(S)
    @test Spad == A[:,:,2]
end
