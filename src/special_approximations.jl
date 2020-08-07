# Coefficients calculated with https://github.com/simonbyrne/Remez.jl

@inline function approx_sin8(x::Union{T,_Vec{<:Any,T}}) where {T <: Real}
    # poly(x) ≈ (xʳ = sqrt(x); sin((xʳ*π)/2)/xʳ)
    x² = vmul(x, x)
    c0 = T(2.22144146907918312350794048535203995923494010677251491220479906920966593121882)
    c1 = T(-0.9135311874994298224944388934705417261765270518848695099428083902179199377101094)
    c2 = T(0.1127023928584587596980569269678174942915399051122642981118394498722218063783927)
    c3 = T(-0.006621000193853498898990183110992108352486751535892362800909323879419896057043918)
    c4 = T(0.0002268980994233557245363541171760472387529757765245978583128895641498725296271051)
    c5 = T(-5.089532691384021959110856232473979525292167742059549332987900223626864039349914e-06)
    c6 = T(8.049906344315649609313027324977744156866597923196983008950128144505665619892402e-08)
    c7 = T(-9.453796623737636858301034347145347814693537235132105505794304057287442064404052e-10)
    c8 = T(8.320735422342537824261297491878000532726851750329165722059039816086266315937799e-12)
    p = vfmadd(vfmadd(vfmadd(
        vfmadd(vfmadd(vfmadd(
            vfmadd(vfmadd(
                c8, x², c7),
            x², c6), x², c5), x², c4),
        x², c3), x², c2), x², c1), x², c0)
    vmul(p, x)
end
@inline function approx_sin12(x::Union{T,_Vec{<:Any,T}}) where {T <: Real}
    # poly(x) ≈ (xʳ = sqrt(x); sin((xʳ*π)/2)/xʳ)
    x² = vmul(x, x)
    c0 = T(1.570796326794896619231321691639751442087433306473273974291471596002143089408967)
    c1 = T(-0.6459640975062462536557565638714840878228221616991079162636960728776200926827234)
    c2 = T(0.07969262624616704512050554673779356754386556916433562280307069303594093234088455)
    c3 = T(-0.004681754135318688100685379129717344612020387136900677528796079362820120586116841)
    c4 = T(0.0001604411847873598218714490487175497697128038098258443135990371062725913378293313)
    c5 = T(-3.598843235212085330760986854968071769532101571696000087022804157171548649851493e-06)
    c6 = T(5.692172921967922014173914534767998233640265149949183614622512228877789263783426e-08)
    c7 = T(-6.688035109809916561166255796732305220204238240380007953680372148880958984800929e-10)
    c8 = T(6.066935730769290440108765932783579488881358689623800108082605751157383050893581e-12)
    c9 = T(-4.377065417731331420103035981625834800685920271728367262547966148291799276629881e-14)
    c10 = T(2.571418016198708615875917881136145309875324227233510421118020948686662677979356e-16)
    c11 = T(-1.253592449512705798908955136513569509617634496103293074276456663908804526347008e-18)
    c12 = T(5.044383456268885650704416950405914330732446213362030631794638992715099861013542e-21)
    p = vfmadd(vfmadd(vfmadd(
        vfmadd(vfmadd(vfmadd(
            vfmadd(vfmadd(vfmadd(
                vfmadd(vfmadd(vfmadd(
                    c12, x², c11), x², c10),
                x², c9), x², c8), x², c7),
            x², c6), x², c5), x², c4),
        x², c3), x², c2), x², c1), x², c0)
    vmul(p, x)
end

@inline suboneopenconst(::Type{Float32}) = 1.9999999f0
@inline suboneopenconst(::Type{Float64}) = 1.9999999999999998
@inline function randsincos(u, ::Type{T}) where {T}
    # return SLEEFPirates.sincos(mask(u, T))
    r = mask(u, T)
    ooc = oneopenconst(T)
    sininput = vsub(r, ooc)
    s = vcopysign(approx_sin8(sininput), u)
    cosinput = vfnmadd(ooc, r, suboneopenconst(T))
    c = vcopysign( approx_sin8(cosinput), SIMDPirates.vleft_bitshift( u, 1 ) )
    s, c
end


@inline function log12_7(x) # each extra coef cuts max error by about 6.5
    c0 = -3.245537891437475818527978529229908008038541532632077901681793316955253799627853
    c1 = 7.133969761783261596308839380142764345357825207927592180328308957632143143761325
    c2 = -7.494130176731051024066811749217581442825262641998555203442250312457063219780548
    c3 = 5.781439120377507449162563050101602606694402138734039600998751617511391254917479
    c4 = -2.985021102358628224549723815823770715482850258619037789336465429092098006063079
    c5 = 0.9780948488102108131759607721708499684604503249092972679349980253346585174787013
    c6 = -0.1839396423223307845519460189847030764334420832555490882187076088746804014578904
    c7 = 0.01512535916800840093163249616452966347663616377265149854210868241945849663079766
    vfmadd(vfmadd(vfmadd(vfmadd(vfmadd(vfmadd(vfmadd(c7, x, c6), x, c5), x, c4), x, c3), x, c2), x, c1), x, c0)
end

# @inline function log12_8(x)
#     c0 = -3.425393083666334067790108512545303457895869147629969342312722110773152768787745
#     c1 = 8.154804072298105118614242101352098813998371180480781803351886784650194633423925
#     c2 = -10.00713525074289150060544073624417642119577303442852729267703418853804728270245
#     c3 = 9.285983673564252215780879208501085478907382239737051965242871871829356007506347
#     c4 = -6.013432610610964086674378270192946254591450512746857671796983783112863051993776
#     c5 = 2.638773868907455181856795850097739510981561473919572777387590013143564751620262
#     c6 = -0.7483574222662208128858646256056442436988142359531261888282163941214837199890769
#     c7 = 0.1238457029355639890633309483614995366456841215797361229753033357177669422002932
#     c8 = -0.009088908104546665130978765013488165959525861872199676157403513022150601871065725
#     vfmadd(vfmadd(vfmadd(vfmadd(vfmadd(vfmadd(vfmadd(vfmadd(c8, x, c7), x, c6), x, c5), x, c4), x, c3), x, c2), x, c1), x, c0)
# end

@inline function log12_9(x)
    c0 = -3.585298173957385989978703196069874884252520373044145691918937267709490632531495
    c1 = 9.175512833125597682830796241005729674265545403887738180485766461445305540243307
    c2 = -12.88095295153181316823685136857741572991739006443899617826355360043323954685944
    c3 = 13.97023726212960122584115270603499243664357604160987633843608122694259681596965
    c4 = -10.88490936111543295028953131289531668167628753782395329546272132276910097503374
    c5 = 5.991068536274933212572646135712744769742009519680374067042945152617478421745127
    c6 = -2.274948730892808615344167059260983651564370908737772545840737790860551065669143
    c7 = 0.5675102681955644633507049003267428212647262631800341022730034095800899973034826
    c8 = -0.0837672704777144218198482121635843275577262111753625917138697703746721702699812
    c9 = 0.005547594786690080999583023292516375924224904367437611790985685156038375417640469
    vfmadd(vfmadd(vfmadd(vfmadd(vfmadd(vfmadd(vfmadd(vfmadd(vfmadd(c9, x, c8), x, c7), x, c6), x, c5), x, c4), x, c3), x, c2), x, c1), x, c0)
end

@inline function log12_16(x)
    c0 = -4.37978460046975823625062481303177042964655415404536866782931434738284915486727
    c1 = 16.31871050954055333956396222100764137104164259191322821181471405491328356924757
    c2 = -43.09849769930020991569153712762384436449878854640277987997630267083267961690192
    c3 = 94.07576064264437747981356599549555160319548024066792684545068563206644131393042
    c4 = -160.2802910490630628148404579429898607373312993988526153520977924333905240690591
    c5 = 214.2893836574384742952624060466181446104100940691092327871064519814161128088174
    c6 = -227.1155113171482030403928820851457771576391114870681368560167920657167422599979
    c7 = 192.2033142217916816132668362174888403726852676823685986312496157227978471900725
    c8 = -130.2749105774541912642407178793062852794871781993373055603363370412151378757816
    c9 = 70.61291078124372952080517981043356347605704563698401990419283029704191724347773
    c10 = -30.40547369086585240971017001009032050577317283007958566465348067875074495409737
    c11 = 10.26725226239003763659962692748550577872238983137251963601200333368657729640871
    c12 = -2.660799790256174577955863634381340009240338534864386807650530364213169177140881
    c13 = 0.5109263929851040905835535339027051835170022363218384968860233384732631558306201
    c14 = -0.0684866374459784970521584206556922390960261357021144271918860042253780684140548
    c15 = 0.005721132779491026341043740576624751586837718238326317641335022781958996821344917
    c16 = -0.0002242388100013828616223303469764320113892766876681769892427408765524456008665568
end

@inline function log12_5_5(x)
    n0 = -6.109900562053389599719353325364255967820553763613149427966199577373249418751788
    n1 = -47.21033890517859950528344185143337802180242634918153480624598815718145572048678
    n2 = -23.72899742183309520410030528859144026073452004231769448510028309291765826686717
    n3 = 52.7756604934258744605387726053489962661521090040028567523770922726855897228027
    n4 = 23.016711670393619896797584282375610751405740717936585553335237271483200924596
    n5 = 1.256864725247499354645915543743422456498233774923938832688724623689696616505147
    
    d0 = 1.0
    d1 = 17.89075270582490509520473828435679777182649663126891381927673222776567771555027
    d2 = 50.90705741116772592703914300524183279313383357917248215823723397522708772511337
    d3 = 35.99672550568958966594388544740561745244861045550134035742275301908396910863221
    d4 = 6.325336279410182109220032402097269110224428178264585067800799249731808867726456
    d5 = 0.1767766952966368811002110905496910114147496513674936569539785785207122898660294

    n = vfmadd(n5, x, n4)
    d = vfmadd(d5, x, d4)
    # n = vfmadd(n, x, n4)
    # d = vfmadd(d, x, d4)
    n = vfmadd(n, x, n3)
    d = vfmadd(d, x, d3)
    n = vfmadd(n, x, n2)
    d = vfmadd(d, x, d2)
    n = vfmadd(n, x, n1)
    d = vfmadd(d, x, d1)
    n = vfmadd(n, x, n0)
    d = vfmadd(d, x, d0)
    vfdiv(n, d)
end

# unlikely to do anything, but avoids bias when more than 12 of the leading bits are 0
@inline function shift_excess_zeros(u::Vec{W,UInt64}, lz::Vec{W,UInt64}) where {W}
    lzsub = vsub(vreinterpret(Vec{W,Int64}, lz), 11)
    bitmask = SIMDPirates.vgreater(lzsub, 0)
    lzshift = vreinterpret(Vec{W,UInt64}, lzsub)
    vifelse(bitmask, SIMDPirates.vleft_bitshift( u, lzshift ), u)

    # lzsub = vsub(vreinterpret(Vec{W,Int64}, lz), 11)
    # lzshift = vreinterpret(Vec{W,UInt64}, SIMDPirates.vmax(lzsub, SIMDPirates.vzero(Vec{W,Int64})))
    # SIMDPirates.vleft_bitshift( u, lzshift )
end
@inline function shift_excess_zerosv2(u::Vec{W,UInt64}, lz::Vec{W,UInt64}) where {W}
    lzsub = vsub(vreinterpret(Vec{W,Int64}, lz), 11)
    SIMDPirates.vleft_bitshift( u, vreinterpret(Vec{W,UInt64}, SIMDPirates.vmax( SIMDPirates.vzero(Vec{W,Int64}), lzsub ) ))
end

@inline function nlog01v2(u::Vec{W,UInt64}, ::Type{Float64}) where {W}
    lz = SIMDPirates.vleading_zeros( u )
    # f = mask(u, Float64) # shift by lz
    f = mask(shift_excess_zeros(u, lz), Float64) # shift by lz
    # l2h = log12_9(f)
    l2h = log12_5_5(f)
    l2 = vsub(l2h, vadd(lz, 1))
    vmul(-0.6931471805599453, l2)
end

@static if Base.libllvm_version < v"8"
    @generated function log2_3q(v::Vec{W,Float64}, e::Vec{W,Float64}) where {W}
        onev = "<double " * join((1.0 for _ ∈ 1:W), ", double ") * ">"
        constv = x -> "<$W x double> <double " * join((x for _ ∈ 1:W), ", double ") * ">"
        constvnotyp = x -> "<double " * join((x for _ ∈ 1:W), ", double ") * ">"
        const1 = constv(reinterpret(Float64,0x3FCC501739F17BA9))
        const2 = constv(reinterpret(Float64,0x3FCC2B7A962850E9))
        const3 = constv(reinterpret(Float64,0x3FD0CAAEEB877481))
        const4 = constv(reinterpret(Float64,0x3FD484AC6A7CB2DD))
        const5 = constv(reinterpret(Float64,0x3FDA617636C2C254))
        const6 = constv(reinterpret(Float64,0x3FE2776C50E7EDE9))
        const7 = constv(reinterpret(Float64,0x3FEEC709DC3A07B2))
        const8 = constvnotyp(reinterpret(Float64,0x40071547652B82FE))
        fma = "<$W x double> @llvm.fmuladd.v$(W)f64"
        decl = "declare $fma(<$W x double>, <$W x double>, <$W x double>)"
        instr = """
          %m1 = fmul <$W x double> %0, %0
          %fma1 = tail call $fma(<$W x double> %m1, $const1, $const2)
          %fma2 = tail call $fma(<$W x double> %fma1, <$W x double> %m1, $const3)
          %fma3 = tail call $fma(<$W x double> %fma2, <$W x double> %m1, $const4)
          %fma4 = tail call $fma(<$W x double> %fma3, <$W x double> %m1, $const5)
          %fma5 = tail call $fma(<$W x double> %fma4, <$W x double> %m1, $const6)
          %fma6 = tail call $fma(<$W x double> %fma5, <$W x double> %m1, $const7)
          %m2 = fmul <$W x double> %0, $const8
          %s1 = fsub fast <$W x double> zeroinitializer, %m2
          %fma7 = tail call $fma(<$W x double> %0, <$W x double> $const8, <$W x double> %s1)
          %a1 = fadd <$W x double> %1, %m2
          %s2 = fsub <$W x double> %1, %a1
          %a2 = fadd <$W x double> %m2, %s2
          %a3 = fadd <$W x double> %fma7, %a2
          %m3 = fmul <$W x double> %0, %m1
          %a4 = fadd <$W x double> %a1, %a3
          %retv = tail call $fma(<$W x double> %fma6, <$W x double> %m3, <$W x double> %a4)
          ret <$W x double> %retv
        """
        quote
            $(Expr(:meta,:inline))
            Base.llvmcall(($decl, $instr), Vec{$W,Float64}, Tuple{Vec{$W,Float64},Vec{$W,Float64}}, v, e)
        end
    end
else
    @generated function log2_3q(v::Vec{W,Float64}, e::Vec{W,Float64}) where {W}
        onev = "<double " * join((1.0 for _ ∈ 1:W), ", double ") * ">"
        constv = x -> "<$W x double> <double " * join((x for _ ∈ 1:W), ", double ") * ">"
        constvnotyp = x -> "<double " * join((x for _ ∈ 1:W), ", double ") * ">"
        const1 = constv(reinterpret(Float64,0x3FCC501739F17BA9))
        const2 = constv(reinterpret(Float64,0x3FCC2B7A962850E9))
        const3 = constv(reinterpret(Float64,0x3FD0CAAEEB877481))
        const4 = constv(reinterpret(Float64,0x3FD484AC6A7CB2DD))
        const5 = constv(reinterpret(Float64,0x3FDA617636C2C254))
        const6 = constv(reinterpret(Float64,0x3FE2776C50E7EDE9))
        const7 = constv(reinterpret(Float64,0x3FEEC709DC3A07B2))
        const8 = constvnotyp(reinterpret(Float64,0x40071547652B82FE))
        fma = "<$W x double> @llvm.fmuladd.v$(W)f64"
        decl = "declare $fma(<$W x double>, <$W x double>, <$W x double>)"
        instr = """
          %m1 = fmul <$W x double> %0, %0
          %fma1 = tail call $fma(<$W x double> %m1, $const1, $const2)
          %fma2 = tail call $fma(<$W x double> %fma1, <$W x double> %m1, $const3)
          %fma3 = tail call $fma(<$W x double> %fma2, <$W x double> %m1, $const4)
          %fma4 = tail call $fma(<$W x double> %fma3, <$W x double> %m1, $const5)
          %fma5 = tail call $fma(<$W x double> %fma4, <$W x double> %m1, $const6)
          %fma6 = tail call $fma(<$W x double> %fma5, <$W x double> %m1, $const7)
          %m2 = fmul <$W x double> %0, $const8
          %s1 = fneg <$W x double> %m2
          %fma7 = tail call $fma(<$W x double> %0, <$W x double> $const8, <$W x double> %s1)
          %a1 = fadd <$W x double> %1, %m2
          %s2 = fsub <$W x double> %1, %a1
          %a2 = fadd <$W x double> %m2, %s2
          %a3 = fadd <$W x double> %fma7, %a2
          %m3 = fmul <$W x double> %0, %m1
          %a4 = fadd <$W x double> %a1, %a3
          %retv = tail call $fma(<$W x double> %fma6, <$W x double> %m3, <$W x double> %a4)
          ret <$W x double> %retv
        """
        quote
            $(Expr(:meta,:inline))
            Base.llvmcall(($decl, $instr), Vec{$W,Float64}, Tuple{Vec{$W,Float64},Vec{$W,Float64}}, v, e)
        end
    end
end
@inline function nlog01(u::Vec{W,UInt64}, ::Type{T}) where {W,T}
    lz = SIMDPirates.vleading_zeros( u )
    # f = mask(u, Float64) # shift by lz
    # f = vmul(0.75, mask(shift_excess_zeros(u, lz), Float64)) # shift by lz
    # f = vfdiv(vsub(f, 1.0), vadd(f, 1.0))
    f = mask(shift_excess_zeros(u, lz), T) # shift by lz
    f = vfdiv(vsub(f, T(1.3333333333333333)), vadd(f, T(1.3333333333333333)))
    # l2h = log12_9(f)
    l2 = log2_3q(f, vsub(T(-0.5849625007211561814537389439478165087598144076924810604557526545410982277943579), lz))
    vmul(T(-0.6931471805599453), l2)
end

