#include <catch2/catch.hpp>
#include <utils.hpp>
#include <vector>

using namespace ss::util;
TEST_CASE("Tests for dot product", "[single-file]") {

        SECTION("int dot_prod_cb_cb_enc(const uint64_t*, const uint64_t*, size_t), dim=64") {
                {
              uint64_t va[] = { 17817334990249652332u }  ;
              uint64_t vb[] = { 14493070588140266406u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 2363268908574242473u }  ;
              uint64_t vb[] = { 11204898385195062253u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 2245626846163845792u }  ;
              uint64_t vb[] = { 7947013789454116966u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 9913451396604795100u }  ;
              uint64_t vb[] = { 11941556533727057263u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 10889409953022963934u }  ;
              uint64_t vb[] = { 7453009400136788415u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint64_t*, const uint64_t*, size_t), dim=128") {
                {
              uint64_t va[] = { 15922672795582195589u,17867185452512421376u }  ;
              uint64_t vb[] = { 4889953930273784654u,10133763144040559341u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 14882216381573871017u,5083964079352261872u }  ;
              uint64_t vb[] = { 7077730751410191287u,8230092248097945631u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -14 );
        }
                {
              uint64_t va[] = { 6883199462028911082u,2100546714401787374u }  ;
              uint64_t vb[] = { 6691160943551794613u,12189453229508109048u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 14797571129337321696u,13349158731945659714u }  ;
              uint64_t vb[] = { 3769556488901609439u,10472441355981240296u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 12088695380165717665u,4988880694995108449u }  ;
              uint64_t vb[] = { 11701329214260809326u,14407028071484732444u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  16 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint64_t*, const uint64_t*, size_t), dim=256") {
                {
              uint64_t va[] = { 9349092695810877953u,11326557229332275103u,14886989203725341159u,16246227854754248752u }  ;
              uint64_t vb[] = { 14128721631740090836u,12425422566535807016u,17831666111319385489u,417724995859793624u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -30 );
        }
                {
              uint64_t va[] = { 12659081133533281008u,13544361449931255448u,12120514576416699576u,1982476507825365715u }  ;
              uint64_t vb[] = { 9910275445221151913u,110058462618715513u,87552661824306995u,970861328919364038u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 13815251953044317115u,6631577990459568517u,14239084848414063664u,1361705026042569827u }  ;
              uint64_t vb[] = { 6594624758285723645u,16590052665205370136u,10237924507423612828u,1535490638169091743u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  8 );
        }
                {
              uint64_t va[] = { 15114342379073475962u,13213936062376731290u,7119210839742890171u,1924628689162420873u }  ;
              uint64_t vb[] = { 8687789191667483235u,6027244009386918710u,17409418675528170220u,15720628752078159694u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  14 );
        }
                {
              uint64_t va[] = { 10502989040901178063u,16050803451218746450u,2979471975136039603u,14419570508978692894u }  ;
              uint64_t vb[] = { 9849869338019904121u,13435711633950126719u,11118076256679667846u,9045687685180162519u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint64_t*, const uint64_t*, size_t), dim=60") {
                {
              uint64_t va[] = { 331796665801776688u }  ;
              uint64_t vb[] = { 693875433961347228u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 331703958729465998u }  ;
              uint64_t vb[] = { 763750295159450712u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 155103465662046925u }  ;
              uint64_t vb[] = { 441975570044025492u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 980901130258343288u }  ;
              uint64_t vb[] = { 132175249817361681u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 465831113445844349u }  ;
              uint64_t vb[] = { 538320836402893662u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint64_t*, const uint64_t*, size_t), dim=59") {
                {
              uint64_t va[] = { 439918962995532417u }  ;
              uint64_t vb[] = { 533314557176076568u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  5 );
        }
                {
              uint64_t va[] = { 371260128548796039u }  ;
              uint64_t vb[] = { 84408964519067733u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -3 );
        }
                {
              uint64_t va[] = { 65812919092834550u }  ;
              uint64_t vb[] = { 574052845516226838u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 76048631752830398u }  ;
              uint64_t vb[] = { 479694652261645747u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -1 );
        }
                {
              uint64_t va[] = { 298579304329635815u }  ;
              uint64_t vb[] = { 248732969952936385u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint64_t*, const uint64_t*, size_t), dim=100") {
                {
              uint64_t va[] = { 8811834993274443726u,54120070843u }  ;
              uint64_t vb[] = { 15995827761072442634u,49100644332u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -12 );
        }
                {
              uint64_t va[] = { 1425996836155922302u,12881028475u }  ;
              uint64_t vb[] = { 2435155716251148321u,68696953640u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 12629907810586402109u,55368678234u }  ;
              uint64_t vb[] = { 6660345350312528116u,9741748428u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 9912277327884906490u,48019358576u }  ;
              uint64_t vb[] = { 2884266280204089075u,53035076776u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 6585792020012519802u,22464135138u }  ;
              uint64_t vb[] = { 12283010940108367053u,23228582382u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint64_t*, const uint64_t*, size_t), dim=119") {
                {
              uint64_t va[] = { 14687867860396484801u,1198004728279799u }  ;
              uint64_t vb[] = { 8922718462204432377u,21101192585712543u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  11 );
        }
                {
              uint64_t va[] = { 7024861214700638585u,19768110653016928u }  ;
              uint64_t vb[] = { 18300665535360926002u,18550541709387775u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  3 );
        }
                {
              uint64_t va[] = { 1260418260884658985u,30603465380237624u }  ;
              uint64_t vb[] = { 8782471341006320799u,8778611621974446u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -5 );
        }
                {
              uint64_t va[] = { 11905846138679354844u,14397210697912109u }  ;
              uint64_t vb[] = { 1807644835825731467u,5083105790298071u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  17 );
        }
                {
              uint64_t va[] = { 11686771343301599681u,11695400590762196u }  ;
              uint64_t vb[] = { 3159246406301811374u,2924447024865809u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  11 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint64_t*, const uint64_t*, size_t), dim=24") {
                {
              uint64_t va[] = { 3712544u }  ;
              uint64_t vb[] = { 16715959u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 12085038u }  ;
              uint64_t vb[] = { 1431915u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 10647232u }  ;
              uint64_t vb[] = { 4658576u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 2726355u }  ;
              uint64_t vb[] = { 11109907u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  8 );
        }
                {
              uint64_t va[] = { 3873498u }  ;
              uint64_t vb[] = { 11690960u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint64_t*, const uint64_t*, size_t), dim=23") {
                {
              uint64_t va[] = { 621681u }  ;
              uint64_t vb[] = { 5602011u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  3 );
        }
                {
              uint64_t va[] = { 1539426u }  ;
              uint64_t vb[] = { 7938710u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -9 );
        }
                {
              uint64_t va[] = { 7848040u }  ;
              uint64_t vb[] = { 6271823u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 3513830u }  ;
              uint64_t vb[] = { 2403504u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  5 );
        }
                {
              uint64_t va[] = { 7574674u }  ;
              uint64_t vb[] = { 6001143u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  7 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint64_t*, const uint64_t*, size_t), dim=150") {
                {
              uint64_t va[] = { 7990865247918961749u,9387547577796329165u,3243562u }  ;
              uint64_t vb[] = { 4444281938371985417u,13773520761862107791u,1111012u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  18 );
        }
                {
              uint64_t va[] = { 2326975449081228106u,13149417657981391175u,2016598u }  ;
              uint64_t vb[] = { 4154917799886313178u,9433133646205203167u,3123105u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 6058273818432852745u,6566156522988959743u,4187961u }  ;
              uint64_t vb[] = { 15437752885919656359u,2139993729288610320u,2673674u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 4367988489714534817u,499414166090573123u,3975183u }  ;
              uint64_t vb[] = { 5991575007446964055u,2018056002004045599u,3786596u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -32 );
        }
                {
              uint64_t va[] = { 12108226222597797314u,11222891623800664401u,4021557u }  ;
              uint64_t vb[] = { 3838388400812028941u,15194836194119699687u,1374911u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -14 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint32_t*, const uint32_t*, size_t), , dim=64") {
                {
              uint32_t va[] = { 1328046406u,1865022384u }  ;
              uint32_t vb[] = { 1216790188u,2472255760u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 803717598u,1395221858u }  ;
              uint32_t vb[] = { 486716074u,4052573155u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 1176328634u,3399844405u }  ;
              uint32_t vb[] = { 3329567684u,1454083983u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 1874147470u,409461909u }  ;
              uint32_t vb[] = { 3485287228u,21058752u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  12 );
        }
                {
              uint32_t va[] = { 3135107154u,1486716254u }  ;
              uint32_t vb[] = { 1612637624u,2786532961u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint32_t*, const uint32_t*, size_t), , dim=128") {
                {
              uint32_t va[] = { 2014113302u,2286550633u,2178094447u,1152506132u }  ;
              uint32_t vb[] = { 376420725u,2036814452u,1695916849u,933057654u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 4010112124u,4005620856u,2459997933u,1409397219u }  ;
              uint32_t vb[] = { 221156136u,2851579109u,507122502u,2680118962u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 1253934618u,2258526217u,1529310626u,2590659185u }  ;
              uint32_t vb[] = { 2959538312u,589938926u,4016327690u,2347434241u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 946517890u,3060834122u,2725531016u,796886118u }  ;
              uint32_t vb[] = { 2629474685u,2340074208u,2665025712u,3318741452u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 3955790618u,1903753070u,1846253494u,673066293u }  ;
              uint32_t vb[] = { 836193587u,3898636097u,1690597415u,615684750u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint32_t*, const uint32_t*, size_t), , dim=256") {
                {
              uint32_t va[] = { 2541750776u,1857272798u,4196974066u,3040068867u,680681040u,1624191000u,1796049483u,4086474020u }  ;
              uint32_t vb[] = { 2194859461u,2127757227u,1412508014u,2458254452u,3581742393u,364442823u,2227576868u,2943809692u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -30 );
        }
                {
              uint32_t va[] = { 1705526615u,2136023779u,1246515156u,1577543624u,2484561288u,4125969625u,854135007u,1407168057u }  ;
              uint32_t vb[] = { 147588782u,3712451067u,1703566558u,3523665936u,3076921942u,1286532266u,207424509u,2528911066u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  14 );
        }
                {
              uint32_t va[] = { 3610377214u,2118252915u,1899729417u,1591820244u,1417306692u,942379658u,3521057067u,3729031712u }  ;
              uint32_t vb[] = { 3751473928u,1776913638u,1162852493u,3784730156u,2554984526u,3735465427u,1767228213u,492538299u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 3228791590u,568878909u,1899273770u,523403407u,2800040183u,3471012772u,1782419720u,3878266848u }  ;
              uint32_t vb[] = { 1472634946u,1088514232u,62194903u,2083274114u,839442138u,2933430748u,2034688276u,4133157887u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 3576362444u,757625606u,3917451529u,1548446155u,1504362513u,3739864035u,1324254762u,1051219485u }  ;
              uint32_t vb[] = { 3205374952u,1505510118u,2843494514u,1215489017u,3247910255u,3443789883u,358746734u,4136231734u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  22 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint32_t*, const uint32_t*, size_t), , dim=60") {
                {
              uint32_t va[] = { 348103803u,56104188u }  ;
              uint32_t vb[] = { 1437570882u,16987910u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 2848077972u,7155108u }  ;
              uint32_t vb[] = { 690451917u,121382554u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 2450649701u,202478957u }  ;
              uint32_t vb[] = { 663544111u,250092111u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 3206570616u,71406118u }  ;
              uint32_t vb[] = { 2810271802u,31379304u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 3886477846u,115171446u }  ;
              uint32_t vb[] = { 28052784u,148220207u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint32_t*, const uint32_t*, size_t), , dim=59") {
                {
              uint32_t va[] = { 2915945052u,93459003u }  ;
              uint32_t vb[] = { 760094357u,90678537u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  5 );
        }
                {
              uint32_t va[] = { 3111526448u,35227831u }  ;
              uint32_t vb[] = { 725003828u,35488716u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  21 );
        }
                {
              uint32_t va[] = { 3612958644u,125273112u }  ;
              uint32_t vb[] = { 2535269590u,9790468u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  9 );
        }
                {
              uint32_t va[] = { 2160054563u,100384125u }  ;
              uint32_t vb[] = { 102932622u,118048634u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -5 );
        }
                {
              uint32_t va[] = { 3224781247u,22617451u }  ;
              uint32_t vb[] = { 3584235532u,15240591u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  5 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint32_t*, const uint32_t*, size_t), , dim=100") {
                {
              uint32_t va[] = { 1284395763u,2960372942u,169356405u,3u }  ;
              uint32_t vb[] = { 3811757109u,1178642465u,2408335298u,5u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -18 );
        }
                {
              uint32_t va[] = { 672148228u,4064865000u,1403493931u,13u }  ;
              uint32_t vb[] = { 2871899977u,1476473813u,2727871017u,10u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 609197605u,2594705121u,1525888406u,12u }  ;
              uint32_t vb[] = { 3030490512u,231976087u,74097933u,15u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 608495320u,320261501u,645774331u,0u }  ;
              uint32_t vb[] = { 1215500307u,1182114773u,154874212u,14u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -12 );
        }
                {
              uint32_t va[] = { 198883352u,2660040595u,3496405551u,9u }  ;
              uint32_t vb[] = { 3617349491u,1651570459u,4031469666u,15u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint32_t*, const uint32_t*, size_t), , dim=119") {
                {
              uint32_t va[] = { 4005434306u,2111085001u,3146005125u,4203049u }  ;
              uint32_t vb[] = { 2762989230u,1278628613u,3195104529u,5024329u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  17 );
        }
                {
              uint32_t va[] = { 2460401990u,2772588643u,3776548682u,163058u }  ;
              uint32_t vb[] = { 1617418032u,3796383443u,1596702161u,7659490u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  5 );
        }
                {
              uint32_t va[] = { 2061630516u,3317155804u,3357219722u,2363257u }  ;
              uint32_t vb[] = { 330690922u,4214873039u,1414912619u,1445819u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  17 );
        }
                {
              uint32_t va[] = { 3125944109u,1094124401u,2581171414u,8002362u }  ;
              uint32_t vb[] = { 379215724u,2528988868u,4282610296u,6376374u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  5 );
        }
                {
              uint32_t va[] = { 2465357131u,293081862u,1131710141u,4692298u }  ;
              uint32_t vb[] = { 3520053924u,3645687142u,909414259u,5050537u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  5 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint32_t*, const uint32_t*, size_t), , dim=24") {
                {
              uint32_t va[] = { 15279718u }  ;
              uint32_t vb[] = { 11514234u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 6346142u }  ;
              uint32_t vb[] = { 12050964u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 8820331u }  ;
              uint32_t vb[] = { 351251u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 2369396u }  ;
              uint32_t vb[] = { 1880425u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 13129372u }  ;
              uint32_t vb[] = { 10050776u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  10 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint32_t*, const uint32_t*, size_t), , dim=23") {
                {
              uint32_t va[] = { 4203723u }  ;
              uint32_t vb[] = { 4226480u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  3 );
        }
                {
              uint32_t va[] = { 4764400u }  ;
              uint32_t vb[] = { 14499u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  5 );
        }
                {
              uint32_t va[] = { 4406822u }  ;
              uint32_t vb[] = { 5288818u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  3 );
        }
                {
              uint32_t va[] = { 5205290u }  ;
              uint32_t vb[] = { 7612002u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  1 );
        }
                {
              uint32_t va[] = { 8198459u }  ;
              uint32_t vb[] = { 4064117u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  3 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint32_t*, const uint32_t*, size_t), , dim=150") {
                {
              uint32_t va[] = { 3619619470u,3574535300u,1578669522u,1697653150u,3833175u }  ;
              uint32_t vb[] = { 2852987928u,844197527u,59071878u,29873102u,4181254u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 290432686u,1837516879u,3192226321u,2983659461u,1500706u }  ;
              uint32_t vb[] = { 3746185108u,2319224633u,2915996493u,909272144u,391165u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 1031507750u,2952978157u,3177052647u,3722859890u,2210448u }  ;
              uint32_t vb[] = { 2285856955u,3007934407u,963894632u,2996395200u,482440u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 2559542296u,796051330u,376012241u,1273644197u,2422478u }  ;
              uint32_t vb[] = { 2880919409u,2260951198u,1774836105u,3193217744u,1746670u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 305984019u,3058226932u,2795721204u,1061541703u,3294937u }  ;
              uint32_t vb[] = { 232405351u,993077485u,2645508025u,3263417755u,3687961u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
            }

    }