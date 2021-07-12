#include <catch2/catch.hpp>
#include <utils.hpp>
#include <vector>

using namespace ss::util;
TEST_CASE("Tests for dot product", "[single-file]") {

        SECTION("int dot_prod_cb_cb_enc(const uint64_t*, const uint64_t*, size_t), dim=64") {
                {
              uint64_t va[] = { 4001061603254483484u }  ;
              uint64_t vb[] = { 5927576687922561269u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 120188359791109595u }  ;
              uint64_t vb[] = { 938502383337190043u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 17917087419485450174u }  ;
              uint64_t vb[] = { 13267876066059593856u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 13299500360057864173u }  ;
              uint64_t vb[] = { 1067151752522503129u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 11224997285451840515u }  ;
              uint64_t vb[] = { 11636401741167677486u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -12 );
        }
                {
              uint64_t va[] = { 2107964811402037715u }  ;
              uint64_t vb[] = { 4214608318937620579u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 6926810271146661162u }  ;
              uint64_t vb[] = { 10706276362105433236u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 13274970790731576336u }  ;
              uint64_t vb[] = { 17087988833500084949u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 6623969829796078505u }  ;
              uint64_t vb[] = { 14441461190092621271u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 4057542175559353839u }  ;
              uint64_t vb[] = { 7543717234364721806u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 14476291150913252820u }  ;
              uint64_t vb[] = { 7903870171677478484u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  8 );
        }
                {
              uint64_t va[] = { 6314313679378051350u }  ;
              uint64_t vb[] = { 293370017152442009u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -14 );
        }
                {
              uint64_t va[] = { 17864328983485847996u }  ;
              uint64_t vb[] = { 9916627793758389463u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -20 );
        }
                {
              uint64_t va[] = { 7110333639071730122u }  ;
              uint64_t vb[] = { 8320721005443255081u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 3643597413653905499u }  ;
              uint64_t vb[] = { 12935123063117225993u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  12 );
        }
                {
              uint64_t va[] = { 7464078853831335011u }  ;
              uint64_t vb[] = { 230311860360064921u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 12960715230402551958u }  ;
              uint64_t vb[] = { 4528381730322573183u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 2356333062145495588u }  ;
              uint64_t vb[] = { 9973465561800383897u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -14 );
        }
                {
              uint64_t va[] = { 14021692564454827172u }  ;
              uint64_t vb[] = { 15236452203238262258u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 1882982642675402278u }  ;
              uint64_t vb[] = { 17226291809069579262u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 7187712490281072261u }  ;
              uint64_t vb[] = { 15030093699186537018u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 17913621152343650049u }  ;
              uint64_t vb[] = { 11912245132218952630u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 5072534999421542591u }  ;
              uint64_t vb[] = { 2196708690175134137u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 2201684887284840382u }  ;
              uint64_t vb[] = { 4374028122461359506u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 6631068229189058757u }  ;
              uint64_t vb[] = { 17773299433036435907u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  12 );
        }
                {
              uint64_t va[] = { 3184377172314175521u }  ;
              uint64_t vb[] = { 15612041482421895946u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 13789446779389127272u }  ;
              uint64_t vb[] = { 6388814638802508174u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -12 );
        }
                {
              uint64_t va[] = { 17979554568153644894u }  ;
              uint64_t vb[] = { 9233459657613755060u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -18 );
        }
                {
              uint64_t va[] = { 11013233313117454651u }  ;
              uint64_t vb[] = { 5131267660505983658u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 17510962793325520007u }  ;
              uint64_t vb[] = { 9940758644090048705u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 6734050945489036482u }  ;
              uint64_t vb[] = { 4767647484976481608u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 5026802490887948640u }  ;
              uint64_t vb[] = { 12201199788591658932u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -12 );
        }
                {
              uint64_t va[] = { 18326924498847530392u }  ;
              uint64_t vb[] = { 3124400955958841051u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 18190575880117362322u }  ;
              uint64_t vb[] = { 6193302943712203306u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  10 );
        }
                {
              uint64_t va[] = { 9951199008042330207u }  ;
              uint64_t vb[] = { 8042958760033320599u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 8852453486176000290u }  ;
              uint64_t vb[] = { 10105466135563494722u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 11893416263528788804u }  ;
              uint64_t vb[] = { 9233919533444452241u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 10487168510782983974u }  ;
              uint64_t vb[] = { 3229215450972124571u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 6243414289315230651u }  ;
              uint64_t vb[] = { 12773292520438385565u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 17319444908016505164u }  ;
              uint64_t vb[] = { 3789230136686841331u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 9286932393696665797u }  ;
              uint64_t vb[] = { 4079849003039979330u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 8091235091161170476u }  ;
              uint64_t vb[] = { 590310928763669407u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 8329865690541261673u }  ;
              uint64_t vb[] = { 710833702785901469u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 10120184916931479006u }  ;
              uint64_t vb[] = { 4969006426347310456u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 8096809653653446130u }  ;
              uint64_t vb[] = { 4662301823488634237u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 14822855862720089824u }  ;
              uint64_t vb[] = { 16228981368152241350u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 16360652989061306279u }  ;
              uint64_t vb[] = { 7399249677086650751u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 15326168368225341700u }  ;
              uint64_t vb[] = { 1594672729453269573u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 13161496933431693949u }  ;
              uint64_t vb[] = { 7337736216352237192u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 12001022975455728531u }  ;
              uint64_t vb[] = { 2212362386551674206u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 4730159972208753648u }  ;
              uint64_t vb[] = { 9165563235518338590u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -12 );
        }
                {
              uint64_t va[] = { 3847295045983782541u }  ;
              uint64_t vb[] = { 215976707424679527u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 4153107805261677090u }  ;
              uint64_t vb[] = { 12446642626498834521u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 6944757489716753334u }  ;
              uint64_t vb[] = { 268849819355871465u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 8106903022508510268u }  ;
              uint64_t vb[] = { 1384591135468233548u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 3138170144947243265u }  ;
              uint64_t vb[] = { 12073648771806405628u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  10 );
        }
                {
              uint64_t va[] = { 7282134069274951346u }  ;
              uint64_t vb[] = { 14663176937346343851u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 72858751756830086u }  ;
              uint64_t vb[] = { 14012909523156545934u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  14 );
        }
                {
              uint64_t va[] = { 2670363741067264523u }  ;
              uint64_t vb[] = { 3522136426340343035u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 3106754700430547520u }  ;
              uint64_t vb[] = { 14728611069002281494u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 2024929669813865781u }  ;
              uint64_t vb[] = { 16832890052709645589u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 14631645658021085142u }  ;
              uint64_t vb[] = { 13886444669980686403u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 3348459380147792261u }  ;
              uint64_t vb[] = { 7737313900544237498u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 4862248689054680948u }  ;
              uint64_t vb[] = { 1366781424072362717u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 9291354895165410681u }  ;
              uint64_t vb[] = { 312416283006928538u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 9652033270948803231u }  ;
              uint64_t vb[] = { 13121951458400599296u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 10083702704506642033u }  ;
              uint64_t vb[] = { 16776617346941844459u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  8 );
        }
                {
              uint64_t va[] = { 1894010794889074953u }  ;
              uint64_t vb[] = { 9662803259086249586u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 4723425091385208663u }  ;
              uint64_t vb[] = { 719471156106133571u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 11534554214795651284u }  ;
              uint64_t vb[] = { 11190103805727596780u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 1718585988447327595u }  ;
              uint64_t vb[] = { 7258674672683566222u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  8 );
        }
                {
              uint64_t va[] = { 14070680571035546721u }  ;
              uint64_t vb[] = { 14890946762870630162u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 18372935826122524052u }  ;
              uint64_t vb[] = { 12054448167390653016u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 15308742659786578537u }  ;
              uint64_t vb[] = { 17879676246342096191u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 17616678472145188848u }  ;
              uint64_t vb[] = { 3606032270170773u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 17281395497552867328u }  ;
              uint64_t vb[] = { 8384905556121606314u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 3538271371749583783u }  ;
              uint64_t vb[] = { 11216090459106216784u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -16 );
        }
                {
              uint64_t va[] = { 4891802734574589468u }  ;
              uint64_t vb[] = { 5365598196489552361u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 18370737906208341097u }  ;
              uint64_t vb[] = { 2690681636465938198u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 16249485251253687108u }  ;
              uint64_t vb[] = { 10425168857870115746u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 1282656092261395938u }  ;
              uint64_t vb[] = { 14511133340489132576u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  10 );
        }
                {
              uint64_t va[] = { 12879132351931603807u }  ;
              uint64_t vb[] = { 5792590601456395238u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 9486599580699713505u }  ;
              uint64_t vb[] = { 7806807947108150307u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 12051173643079216300u }  ;
              uint64_t vb[] = { 10106986016382973661u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 9807245674874987451u }  ;
              uint64_t vb[] = { 11150006096656133699u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  14 );
        }
                {
              uint64_t va[] = { 17459256995190467618u }  ;
              uint64_t vb[] = { 7779773739444241648u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 2947036162272399547u }  ;
              uint64_t vb[] = { 7465429906471970819u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 18408336680287460417u }  ;
              uint64_t vb[] = { 14458802227125027874u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 6327045898777539237u }  ;
              uint64_t vb[] = { 18445588672808449492u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 11107207829142611319u }  ;
              uint64_t vb[] = { 5096977805421229249u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 871848625340351352u }  ;
              uint64_t vb[] = { 14704327898086869005u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  10 );
        }
                {
              uint64_t va[] = { 1529129064132415874u }  ;
              uint64_t vb[] = { 13341494002004088868u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 9326238200749165783u }  ;
              uint64_t vb[] = { 961720703677262532u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 18071685040570027406u }  ;
              uint64_t vb[] = { 13890823004657805778u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  8 );
        }
                {
              uint64_t va[] = { 10559744124610887222u }  ;
              uint64_t vb[] = { 6057089889146805184u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 4051610111785396051u }  ;
              uint64_t vb[] = { 9595924660664459837u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 666816520400768462u }  ;
              uint64_t vb[] = { 15564925919279985953u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 15179697602913673297u }  ;
              uint64_t vb[] = { 14998956613311299160u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 5859167054334106092u }  ;
              uint64_t vb[] = { 10811844196412769268u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 13287382359360973917u }  ;
              uint64_t vb[] = { 12140139627398997755u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  20 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint64_t*, const uint64_t*, size_t), dim=128") {
                {
              uint64_t va[] = { 7750716248045600741u,2034352381923062712u }  ;
              uint64_t vb[] = { 13790739308459162365u,14808527258590625283u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 18086336786109006258u,6320064746348439936u }  ;
              uint64_t vb[] = { 15605928701727784593u,6348701410438456347u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 2322804637276264177u,14175753979853876206u }  ;
              uint64_t vb[] = { 11099913486951762016u,9379098060947320759u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  14 );
        }
                {
              uint64_t va[] = { 6271886868662685749u,13946447994652030839u }  ;
              uint64_t vb[] = { 17965671814566728608u,7873329198285655103u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 11285270676464699136u,13858623652585860555u }  ;
              uint64_t vb[] = { 1574218259127409028u,5623244146508934063u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 3353762040806632423u,12204544005091549081u }  ;
              uint64_t vb[] = { 1501572316118603821u,9802783276065962924u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 12304326946691433636u,11813676450528463074u }  ;
              uint64_t vb[] = { 17039783393711371811u,15143625229668372345u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 8692667655139718143u,3381598604944273439u }  ;
              uint64_t vb[] = { 12263194780708406060u,2187359624311662765u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 1901694648495556675u,7478570837352600679u }  ;
              uint64_t vb[] = { 8177730654641228977u,2855422291088300260u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 15939923153002735780u,2848664553196337634u }  ;
              uint64_t vb[] = { 11548432527148536552u,14745431844728796850u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 14414678315013661129u,8042300506766677708u }  ;
              uint64_t vb[] = { 6227183368894563620u,13497000328481105614u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 1873913522079886413u,12779428834813319338u }  ;
              uint64_t vb[] = { 12655781027674879162u,4884651658383286773u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 14926591234179712769u,15133799520880429638u }  ;
              uint64_t vb[] = { 15457599992600283011u,15632710473020003795u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  16 );
        }
                {
              uint64_t va[] = { 15224563060819467133u,15178499354648209251u }  ;
              uint64_t vb[] = { 1241763191581378104u,14387183026887469218u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 5105969726922064367u,12738673852340674904u }  ;
              uint64_t vb[] = { 9162896129016010315u,9448856663674432418u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 9189075870107666511u,7614328556860099433u }  ;
              uint64_t vb[] = { 6904298895169052988u,646751292292226357u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 7852448108955138497u,7759191933378388436u }  ;
              uint64_t vb[] = { 16567420493883515472u,11405005641393265254u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  14 );
        }
                {
              uint64_t va[] = { 1314069110327046280u,14358218982804947230u }  ;
              uint64_t vb[] = { 8750648207488127748u,140161741471531411u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  10 );
        }
                {
              uint64_t va[] = { 13453828748817972078u,10141786369616304348u }  ;
              uint64_t vb[] = { 12239926182313700331u,9868946313617220215u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 7673737615640413836u,8458931919364144361u }  ;
              uint64_t vb[] = { 14255490843491066891u,8129029230223573606u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 16966997999409078133u,8154664701743350588u }  ;
              uint64_t vb[] = { 1899756112092644883u,3949532364748292812u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 7245510462886476428u,13737014456610904511u }  ;
              uint64_t vb[] = { 18214417638251814323u,9360042607259311708u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 879744906657228977u,4650714566062861742u }  ;
              uint64_t vb[] = { 10335806668685348701u,12090700587611736650u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 14069974776883098637u,17904780928500305940u }  ;
              uint64_t vb[] = { 8997424764693330056u,6950255121040222835u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 521229640802562037u,15492951614096794019u }  ;
              uint64_t vb[] = { 1804526482984297402u,11147767734600110376u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 4065026637484554276u,7984063807838273395u }  ;
              uint64_t vb[] = { 1667749828311984251u,15469255420680119852u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  12 );
        }
                {
              uint64_t va[] = { 14088883224872167685u,5724375548949737637u }  ;
              uint64_t vb[] = { 315707761701169769u,3848220622331409677u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -12 );
        }
                {
              uint64_t va[] = { 15076541842416767899u,7210205874505652716u }  ;
              uint64_t vb[] = { 4187000202015337752u,2715707536140377704u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 11287398911514232890u,5728086275604681834u }  ;
              uint64_t vb[] = { 144811200281383689u,879173930766067249u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 1213584243079418251u,5259363335574941016u }  ;
              uint64_t vb[] = { 10557979383181259384u,16592725281415753744u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 748037257435733969u,3224292787821497525u }  ;
              uint64_t vb[] = { 13068921386447169978u,18138870650753072887u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 10047949673285083182u,16177959388497856839u }  ;
              uint64_t vb[] = { 6391268271132829289u,6894380427122356821u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  14 );
        }
                {
              uint64_t va[] = { 8955103474448016139u,1113200682450262540u }  ;
              uint64_t vb[] = { 14099206598791681559u,1436131181863180662u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 941347866195849710u,2125865502691541620u }  ;
              uint64_t vb[] = { 11683456980410803314u,15519278406636095863u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 9720126933954375062u,12943097668705514716u }  ;
              uint64_t vb[] = { 2211903134311627464u,10159538963518339124u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  10 );
        }
                {
              uint64_t va[] = { 12633080083999223274u,5155762419180581525u }  ;
              uint64_t vb[] = { 7822359134696267923u,391379010030398339u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 1875906236596148108u,5087705621102953109u }  ;
              uint64_t vb[] = { 9381603060986147933u,2638684001193270515u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  18 );
        }
                {
              uint64_t va[] = { 17968045087656374862u,17035288512765746800u }  ;
              uint64_t vb[] = { 7504216336896444903u,16428934173703792906u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 3462032379594093609u,5450930477564791146u }  ;
              uint64_t vb[] = { 12741828520410388786u,7747930364278362683u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -14 );
        }
                {
              uint64_t va[] = { 16423776305284902831u,3090292266464208190u }  ;
              uint64_t vb[] = { 295138165801952671u,14128165207470817181u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -22 );
        }
                {
              uint64_t va[] = { 13973577786004142044u,14816744224654323067u }  ;
              uint64_t vb[] = { 16361167635958859822u,5858061605349618517u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 8639814120987445577u,9235983866188254261u }  ;
              uint64_t vb[] = { 6834120918635284481u,18246096273674106031u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 2955665655807181273u,8980170770818485637u }  ;
              uint64_t vb[] = { 7063012656572891053u,1376565610363756316u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 13338059995594164170u,4827129388994546699u }  ;
              uint64_t vb[] = { 47678858232358069u,5047986247351909707u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 16370799752086013515u,14126207286973686385u }  ;
              uint64_t vb[] = { 12523918183383752321u,125269910825744142u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -16 );
        }
                {
              uint64_t va[] = { 11984216909243218694u,8708055260856254343u }  ;
              uint64_t vb[] = { 17389431578062001474u,10489291064151495107u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 4164409762079795629u,15400724517042583839u }  ;
              uint64_t vb[] = { 4969457041378525381u,251884794261271523u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 4161064034249274518u,11112510074634860152u }  ;
              uint64_t vb[] = { 14047234309975943965u,130979949762886829u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -30 );
        }
                {
              uint64_t va[] = { 5002533922749800326u,7780371143098132505u }  ;
              uint64_t vb[] = { 6077393007303244264u,3599838675381916856u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 11285653125521748574u,2992447384916395647u }  ;
              uint64_t vb[] = { 3036623881886357979u,17725002754751347455u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 3996734667670377814u,4235956455170889480u }  ;
              uint64_t vb[] = { 10138374250221239114u,5259486360804855648u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 13411430196296688955u,3615933672738476923u }  ;
              uint64_t vb[] = { 5188494880092368338u,16732183061945751554u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 16725864637044903431u,5343543829008306523u }  ;
              uint64_t vb[] = { 9047763314049215909u,10765037144361516366u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 11047699967380031826u,11331313295311782948u }  ;
              uint64_t vb[] = { 16209904013057895898u,11295679085988193941u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  12 );
        }
                {
              uint64_t va[] = { 3788208978114459170u,3447967554986710313u }  ;
              uint64_t vb[] = { 7121398456689701567u,11398538016593539207u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 16662544807305511075u,13839459467572806712u }  ;
              uint64_t vb[] = { 13786165530669875755u,8456157750173263937u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 4719592907579722135u,3907249133734824247u }  ;
              uint64_t vb[] = { 1719105575260982489u,12124726892642031038u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 11629448640101263409u,13453209833781694796u }  ;
              uint64_t vb[] = { 3720401195475797986u,11379948245896722911u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 3256930071681782284u,13479005710362185106u }  ;
              uint64_t vb[] = { 6174678083877327867u,17671510867163296722u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  14 );
        }
                {
              uint64_t va[] = { 8427344951133802638u,6139225894825416548u }  ;
              uint64_t vb[] = { 11902817656261601185u,1398947679605584821u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -14 );
        }
                {
              uint64_t va[] = { 13870326244450017833u,14641423742853113728u }  ;
              uint64_t vb[] = { 11544016257189349540u,13594292772406625346u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  10 );
        }
                {
              uint64_t va[] = { 7545034935228993811u,3931859484184031742u }  ;
              uint64_t vb[] = { 1188939650646291515u,8992767220988917189u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 576541105399463270u,9259423257640922106u }  ;
              uint64_t vb[] = { 12835972916832081593u,17583370921364314819u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 1517214416723721565u,1753228309829804009u }  ;
              uint64_t vb[] = { 9856929393738225077u,17889000947167399224u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  16 );
        }
                {
              uint64_t va[] = { 6794912338400244837u,7322859735618175954u }  ;
              uint64_t vb[] = { 11338118998289005037u,5142002263972727060u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 2599056935977149241u,8380155016677744648u }  ;
              uint64_t vb[] = { 7430870135861386530u,3683133241004268545u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -16 );
        }
                {
              uint64_t va[] = { 13650030651759871211u,18002061711000478751u }  ;
              uint64_t vb[] = { 12944718813798015574u,12382139953579772946u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  16 );
        }
                {
              uint64_t va[] = { 15276750211783628024u,8753594125069314924u }  ;
              uint64_t vb[] = { 9984952033066481619u,13280610079081428142u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 13277471952712440570u,14121365378826932607u }  ;
              uint64_t vb[] = { 100040441123611037u,5509873958598841341u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 2551724784902584366u,15275177340642332045u }  ;
              uint64_t vb[] = { 2880615336663502848u,11736200863240073673u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  14 );
        }
                {
              uint64_t va[] = { 15889909395500538745u,15681420056019229276u }  ;
              uint64_t vb[] = { 66313340301648753u,2780798539595532657u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -14 );
        }
                {
              uint64_t va[] = { 5734935966902929622u,6277743597399006760u }  ;
              uint64_t vb[] = { 1828681405062617694u,4597316527095323408u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 2851225865525977591u,12045933730531311750u }  ;
              uint64_t vb[] = { 7268592874110643318u,3748520454559953982u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 13193107096924691952u,2814179745652115413u }  ;
              uint64_t vb[] = { 4742454518501531910u,5078349255808746227u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  14 );
        }
                {
              uint64_t va[] = { 10327428050171968643u,7821981448058755266u }  ;
              uint64_t vb[] = { 970684600649607708u,5779915561628221938u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 8192637255506517818u,13290940271331428068u }  ;
              uint64_t vb[] = { 17815488146914983021u,11363888519827533226u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  10 );
        }
                {
              uint64_t va[] = { 12742589978171846155u,5416237388655806306u }  ;
              uint64_t vb[] = { 12481076518201078278u,14696720635117864110u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 3456911629220257661u,718760690925293411u }  ;
              uint64_t vb[] = { 9077035859452946295u,11293954564863697612u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 3274439527963698586u,11139909679069029697u }  ;
              uint64_t vb[] = { 8727216815257327121u,10900574923618083800u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 1899317453702303199u,17673711074320845482u }  ;
              uint64_t vb[] = { 16913117634955539243u,2611152576247640230u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 4964840360270896769u,2331526859582729373u }  ;
              uint64_t vb[] = { 1433616579430408604u,10190728797113963388u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -14 );
        }
                {
              uint64_t va[] = { 9270931859418160643u,13181204073824907032u }  ;
              uint64_t vb[] = { 11715909420241221913u,14451210421828196366u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 9757481449737719196u,13618853188138379335u }  ;
              uint64_t vb[] = { 14310300043508006228u,118946807694942483u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 16170817642158919801u,9507642491192641983u }  ;
              uint64_t vb[] = { 3357261301500577241u,634743933554169762u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 13828925569439614421u,6742304885613311409u }  ;
              uint64_t vb[] = { 8935814734092261479u,18441436597580998694u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 11584240444810005585u,2821592203824059070u }  ;
              uint64_t vb[] = { 13797270877466709696u,6857965455371671189u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  16 );
        }
                {
              uint64_t va[] = { 14159979791441050939u,14919660719494676823u }  ;
              uint64_t vb[] = { 13685513558610306530u,12906976986675496467u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 6245092551055319558u,12714077608178600633u }  ;
              uint64_t vb[] = { 11907638404398785182u,12234206223882072378u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 7345703233130372999u,15796463793698977889u }  ;
              uint64_t vb[] = { 18001875031931451381u,1237878197533991564u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 1830719797808218907u,17833947383736031725u }  ;
              uint64_t vb[] = { 15390332340965815136u,1028650814897684251u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -20 );
        }
                {
              uint64_t va[] = { 4473637326874461978u,12379352580802369953u }  ;
              uint64_t vb[] = { 10025911822225586886u,11212093393860438841u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 12762537582963734703u,2186931875841718578u }  ;
              uint64_t vb[] = { 4524866381039355272u,5895179394761816190u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 17212169106038714308u,11160954504690397060u }  ;
              uint64_t vb[] = { 15686079637228199907u,2176983190448709551u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 13949308038560291198u,6369733247824545032u }  ;
              uint64_t vb[] = { 9795421223754904001u,13479118295894439365u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 13656531408689396264u,14701998466907067381u }  ;
              uint64_t vb[] = { 16856838547013058519u,10105266394957065414u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 16332348720964877268u,14890709634626813701u }  ;
              uint64_t vb[] = { 5323488941627990798u,16424862647297255567u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 16867600676060808680u,8412171623862916446u }  ;
              uint64_t vb[] = { 2968662987542780566u,6821099032130827127u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 7564196629624248563u,8307514138370413083u }  ;
              uint64_t vb[] = { 17986229151270706824u,5815187960697575080u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 9765599935718164012u,12649105004084299038u }  ;
              uint64_t vb[] = { 2125498209087306050u,8087960490571912004u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -12 );
        }
                {
              uint64_t va[] = { 13003280631228765881u,13607636100776136772u }  ;
              uint64_t vb[] = { 7290453600246117897u,9589078783410357701u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  10 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint64_t*, const uint64_t*, size_t), dim=256") {
                {
              uint64_t va[] = { 7609970455010582981u,4103817404811840802u,9436428180024578555u,15328782556817325218u }  ;
              uint64_t vb[] = { 3922122568357424586u,12708765083482588922u,7103587160241222263u,14042755589789017782u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 5865738487120505694u,8426581533063588242u,14512121049671678880u,1954870574625988258u }  ;
              uint64_t vb[] = { 1653484142418922323u,4871155605490973474u,943519488180842198u,593139351839778049u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  10 );
        }
                {
              uint64_t va[] = { 515369345880153970u,15890486751942867381u,17697450302778238461u,3617132633236868382u }  ;
              uint64_t vb[] = { 13245420235227365936u,5227884482099622411u,17909347908437074196u,10868849112331997537u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 1816375628858181934u,15933158937057287237u,5332170002331620632u,15901136106864148001u }  ;
              uint64_t vb[] = { 4807023967164778764u,6409823260180319157u,12637574555381161898u,11442570736599022840u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  20 );
        }
                {
              uint64_t va[] = { 76892926120042761u,15613557650293944330u,17499307526738259757u,9124151076245933469u }  ;
              uint64_t vb[] = { 13619520268602171963u,12351228744874688608u,916547912828033064u,7749933595636907644u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  16 );
        }
                {
              uint64_t va[] = { 8596592695034997642u,2023660342292121470u,13676013573534086202u,143838730829785989u }  ;
              uint64_t vb[] = { 14757416782430159574u,9083713322743957775u,14711866757102106739u,1901864296737516038u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  8 );
        }
                {
              uint64_t va[] = { 15174694180259875131u,17371702149909867732u,12644131630668721544u,18043728452732868371u }  ;
              uint64_t vb[] = { 12224932046618817201u,4963041663808982443u,9101305237238633239u,15259106553323893960u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -36 );
        }
                {
              uint64_t va[] = { 3668063292178089103u,11815729224716889289u,7494543304841167104u,16746651887783538532u }  ;
              uint64_t vb[] = { 17539620041706909551u,13025897514015687866u,871704070254063962u,13082039105774883485u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -24 );
        }
                {
              uint64_t va[] = { 2581893942227401134u,7112398502352586480u,13590323929810580876u,17988556598359804314u }  ;
              uint64_t vb[] = { 2835352171478606679u,5733923355582361934u,16642121731990069320u,16447715835549469693u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  10 );
        }
                {
              uint64_t va[] = { 9962315130203937121u,4985649142777573283u,5337903318954556756u,8859136723355927875u }  ;
              uint64_t vb[] = { 8763826580903169207u,10416967816658893023u,847464758564172040u,12814569761550438003u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 6355224606369503383u,11098089200577158496u,8462325809789813175u,12544949796454259321u }  ;
              uint64_t vb[] = { 9870064946259579779u,13692637584179468004u,7647836431607612517u,16268737482753573329u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 4118063598429113930u,18356233470752935302u,16801258855612487643u,10183711591088722694u }  ;
              uint64_t vb[] = { 12654953167777251u,17365497184031062275u,1229402820355466678u,11814472019564842906u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  14 );
        }
                {
              uint64_t va[] = { 6198354127960427002u,11235176946283131278u,15318208755949735281u,18277905377429182100u }  ;
              uint64_t vb[] = { 5106297758007260084u,10799840756856715279u,13961124364137661396u,2234670142379103866u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  12 );
        }
                {
              uint64_t va[] = { 9660158876136897689u,11915654608625198867u,1729379980212658910u,11791637359500826349u }  ;
              uint64_t vb[] = { 14364120390794857603u,17740101082346077627u,14180561816731919170u,1660546170966475917u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  14 );
        }
                {
              uint64_t va[] = { 17794116712851730549u,3706648938825728629u,18112745154429060440u,2108715552443380280u }  ;
              uint64_t vb[] = { 4486877188068339486u,8391354952631734693u,2056323860696355490u,18440195579547252470u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 5268500100037425301u,12812078900992429218u,3111006223742933436u,10494408036568713u }  ;
              uint64_t vb[] = { 8258708740044039941u,16282779140340490558u,4300187539611479238u,2787918238561562890u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 7915851370650378673u,9662059754581732108u,15546561361626038620u,5347155386051457770u }  ;
              uint64_t vb[] = { 3599024091795502975u,6817852819849636678u,18054338313938251243u,8938967582828450500u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 16779092476260079059u,8914156946031518573u,9943094099502147650u,1055089508143353158u }  ;
              uint64_t vb[] = { 4008521433540741876u,14370442918107084264u,2276435284093357068u,1321152226400148114u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -24 );
        }
                {
              uint64_t va[] = { 17430260848867596682u,1212677666201298845u,6117402835287867811u,3658085887311826642u }  ;
              uint64_t vb[] = { 11981107432871574529u,6654146992472203044u,15137190141929382257u,15414235175255808152u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 14086179518934611652u,5827881476190710682u,14624164015387342356u,2937266406652273966u }  ;
              uint64_t vb[] = { 1154425362494796758u,1043953121306699550u,14424241071069707703u,1218091522595262753u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  22 );
        }
                {
              uint64_t va[] = { 13552805740889551521u,7338732001237408238u,3019235679606619696u,1893238992398117775u }  ;
              uint64_t vb[] = { 7002483892743220219u,14891610197511865488u,2315247515199978995u,1246893887350209929u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 3661715090939716075u,14532511326972077356u,15797855301288465302u,5179050847547730823u }  ;
              uint64_t vb[] = { 17526502519811024595u,17575924625194313423u,15997306290386943169u,10604856462874271800u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 6988409752948924425u,11357483526624947660u,10346274856430984477u,10064829223936640574u }  ;
              uint64_t vb[] = { 9799153686350201327u,17124044930455571363u,11052290964167484543u,12252037353514605394u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -24 );
        }
                {
              uint64_t va[] = { 11642171693371981844u,5347574887948253661u,10984996850090826320u,11253823450776871514u }  ;
              uint64_t vb[] = { 12427769444489983460u,690550562041456240u,10834441682700863783u,3787581244839119869u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 3288030392547560380u,16053771639725391246u,2212443845620716267u,7139842292462071300u }  ;
              uint64_t vb[] = { 17802037271718401888u,16268716144537990612u,16612888593401401175u,7040168018059731217u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -12 );
        }
                {
              uint64_t va[] = { 11459662559815355495u,1413094152310144670u,3672864837825787979u,14175536240078737723u }  ;
              uint64_t vb[] = { 16302280519788221575u,7516911359124626864u,413417867044504186u,8878760910386973056u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 1813007008614468157u,18205209056608275032u,1834829607945410549u,16171176539957868157u }  ;
              uint64_t vb[] = { 12579680658400886326u,13914893091533046304u,3703282964373611015u,14413700520729215728u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  22 );
        }
                {
              uint64_t va[] = { 2381249775404179171u,16907432320995714252u,1056242949122934677u,11738804000832388053u }  ;
              uint64_t vb[] = { 12463459926232566582u,16869828784221591753u,14987805556927516723u,9850336002147543672u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -32 );
        }
                {
              uint64_t va[] = { 12164732546196478650u,12800021418228819628u,13556712511480472967u,17464156128936467547u }  ;
              uint64_t vb[] = { 14679669561725273977u,3754714546685261128u,3869125752369505040u,7391412160404488592u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 13885088052206843160u,6936240404913965890u,14951695008709605917u,4950630935303727591u }  ;
              uint64_t vb[] = { 14041042630161079822u,16455685148580366093u,15287427482013831951u,15978138619321067160u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  20 );
        }
                {
              uint64_t va[] = { 6627647923920022266u,3984561963124454643u,16664249806854613893u,6868350164708532106u }  ;
              uint64_t vb[] = { 11058557881986286946u,10841460276142995490u,211599722186176736u,7060273160503721410u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  18 );
        }
                {
              uint64_t va[] = { 5536460884035851246u,5566630212072983087u,5795546789335779906u,3904361805009866871u }  ;
              uint64_t vb[] = { 9928153432084339173u,16323611415083741468u,11393507136380011789u,411295148414046558u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  8 );
        }
                {
              uint64_t va[] = { 3900430133704604414u,18422320184104744902u,5342660439820688057u,211094429451703922u }  ;
              uint64_t vb[] = { 17176658975672318220u,5389291578064392415u,2996415925991934188u,14970545037781065058u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -12 );
        }
                {
              uint64_t va[] = { 12240358337550362605u,13933066826757424409u,14804414638003502311u,13280908121762127674u }  ;
              uint64_t vb[] = { 11437942159971773102u,2075278196559036500u,16313304535634998104u,134727733756893017u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 8736912239551325187u,10677668234814780724u,1799836705727797237u,15391295738263407967u }  ;
              uint64_t vb[] = { 2984199468922147522u,3942765220247267432u,1024155054817350695u,8773400825981767225u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 12196325502501901664u,4156892959442775016u,7677866561069843677u,3307470051265223911u }  ;
              uint64_t vb[] = { 4501013473979872000u,3101176026609580240u,13953102321926876156u,1772795351992180291u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  26 );
        }
                {
              uint64_t va[] = { 15435804302037238558u,6702018291539563527u,9404313419562056025u,530938253065414974u }  ;
              uint64_t vb[] = { 3155018441237125366u,10665275214314952981u,3466236523524201037u,1622731287688371506u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 15387080427747965295u,492281617597025070u,17275564581305169357u,17478498257547289016u }  ;
              uint64_t vb[] = { 8357687638181965476u,5864424860898883930u,15624496880077814248u,8897528373814987052u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -14 );
        }
                {
              uint64_t va[] = { 3971420402590626563u,17976440180723534531u,15692233188012854298u,4746173701552153942u }  ;
              uint64_t vb[] = { 3753925032527487104u,9921566793985821940u,10781839310000539098u,18060218860515791806u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  14 );
        }
                {
              uint64_t va[] = { 3783989778491830068u,6437800916206743069u,5982600618045976463u,5101449713499564354u }  ;
              uint64_t vb[] = { 8827367399447290590u,14298893643642763377u,7191453306428860314u,5341668125120233114u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 10688028613764667935u,5042844685738349464u,12334401272221063199u,12436780747081747343u }  ;
              uint64_t vb[] = { 2722291963013073758u,8846533301190209034u,3451701340924565309u,15861901162916083589u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 4459045243093129259u,4075064636277022880u,11935931635794452566u,10211479693150845165u }  ;
              uint64_t vb[] = { 14186231945940452033u,12538256398048407663u,11396899879682793066u,5859530175906950386u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -28 );
        }
                {
              uint64_t va[] = { 2938162690595866486u,10237496617046344980u,8804294773155167918u,8900162152484366581u }  ;
              uint64_t vb[] = { 6016519540749105845u,1529827772449595628u,8874518708927647542u,1099657833401526948u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 6294088993176810666u,6484789256871225826u,14636730764073468495u,3608087068914955123u }  ;
              uint64_t vb[] = { 3636098374234888523u,9788314859521881598u,4427994403681520891u,15571280068104781217u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -14 );
        }
                {
              uint64_t va[] = { 13889121207749411856u,9028601610899142768u,1090726696384038576u,5967476931327806269u }  ;
              uint64_t vb[] = { 5138250876133699402u,18256007877901686349u,953204724958799873u,10673605230625223103u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  12 );
        }
                {
              uint64_t va[] = { 10549594625909484036u,15333947454143766575u,5629874288795838553u,12322148065956937494u }  ;
              uint64_t vb[] = { 18276712308143414391u,3364425235786044020u,10851245883052798166u,15938509878822271589u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -18 );
        }
                {
              uint64_t va[] = { 7041146125033080563u,2981853151101915524u,2146725038338000414u,18083948796277309301u }  ;
              uint64_t vb[] = { 10880126258052256879u,11840753808507947736u,4961895905656871331u,16732990270362147903u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 16693511439186081091u,12988787795746330029u,2513826971525799105u,4435006615524922605u }  ;
              uint64_t vb[] = { 1174631298174645531u,1527050313214270601u,1112903360170657271u,14276244856143098767u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 1178079768816135054u,202681900329100725u,5215490536111187533u,2974286200604664027u }  ;
              uint64_t vb[] = { 5682052471798907376u,11266606455890704154u,13722974454148054050u,8835479764532287254u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -24 );
        }
                {
              uint64_t va[] = { 15756328430424228698u,10993859029681665856u,11850381607552256835u,3877542918828156023u }  ;
              uint64_t vb[] = { 11313027866251642792u,6866138092976387002u,17726933339921004700u,8516019781146067109u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -20 );
        }
                {
              uint64_t va[] = { 2244653995735699428u,13536874806498837816u,4659657239651999716u,11493162414440286392u }  ;
              uint64_t vb[] = { 6747410248615994926u,17992747060291170053u,11136460321164821193u,8103879741132058551u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -12 );
        }
                {
              uint64_t va[] = { 10930828175383305495u,9251188374248680975u,4993208117614778731u,13107562434855224862u }  ;
              uint64_t vb[] = { 6457954014092402879u,14737036343423049983u,11300000087358330135u,8162844048461399377u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -24 );
        }
                {
              uint64_t va[] = { 1690949908153535594u,6541964541456984491u,11199974555075942187u,1651671250001702487u }  ;
              uint64_t vb[] = { 5447134875865717721u,8499848923089623054u,4048462842788750890u,2824146298268026u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -14 );
        }
                {
              uint64_t va[] = { 12042589719648264016u,7778994786081765136u,13937086975613396783u,1047377215123701346u }  ;
              uint64_t vb[] = { 9445417845596013541u,1001684187613668767u,5126177530010653398u,7010311208897215894u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 7704462997279211175u,9938123007059801407u,9394178184107172256u,14094698845797709099u }  ;
              uint64_t vb[] = { 6504742525208313822u,7364893660551728408u,15608801021473201001u,10532483782364521808u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 9555249404522834829u,6359199352645868707u,5834963940560475597u,4343448630783149507u }  ;
              uint64_t vb[] = { 15486940936431799389u,2867982441238959223u,5199509491425152220u,13469733002869147833u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  8 );
        }
                {
              uint64_t va[] = { 15103204897407309719u,2203282008373101917u,11098818207583604256u,13986498902200034204u }  ;
              uint64_t vb[] = { 4040700401568887998u,10350308497333789925u,3266577582501699038u,12117649875674796284u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 18263390328090155962u,13106831682446893771u,7780073626061911866u,7327265708057205154u }  ;
              uint64_t vb[] = { 2447176354384237422u,14750579039410797724u,17522919503033615336u,14380871886686183119u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  10 );
        }
                {
              uint64_t va[] = { 4762276674029495437u,12398405865938655177u,2681967235396054770u,15357005957033977362u }  ;
              uint64_t vb[] = { 16466026741659190334u,16556273142335515124u,11942024034946793657u,2576679786859470299u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 1867464483444827931u,14124422412859430037u,14143198893245832510u,5165148602310919048u }  ;
              uint64_t vb[] = { 16485089035073072472u,11655648629903236368u,15691051166043840000u,3411206268474820709u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  10 );
        }
                {
              uint64_t va[] = { 4247198244568874087u,2278481703926908212u,14878323489469897193u,8304584721102263984u }  ;
              uint64_t vb[] = { 5000331675980036053u,12151455137018391588u,15872892815549986392u,5368089198307182444u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -26 );
        }
                {
              uint64_t va[] = { 11778604501361316998u,10323760714426701406u,10400347491883774335u,9130682309253683982u }  ;
              uint64_t vb[] = { 4826034533538351109u,9540245487779875257u,10628745742235119556u,8676745173721127317u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 7520264826086863276u,10052996801511338603u,11210515946898844598u,2359321752252400595u }  ;
              uint64_t vb[] = { 16216911654861647216u,15595731385902797749u,18070628736622691467u,3597417339595950839u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  32 );
        }
                {
              uint64_t va[] = { 16210030447954675802u,10829575166535547630u,4216222562703155750u,7291442370175389342u }  ;
              uint64_t vb[] = { 8948716138249454754u,1394599412975418118u,3031808136217920537u,11103809428816532198u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  16 );
        }
                {
              uint64_t va[] = { 8412062914150632750u,4846125228039458960u,14965697599887468391u,16046465100261459037u }  ;
              uint64_t vb[] = { 6090799547853234577u,9038469400914432741u,11487071247528559113u,1683875908124870993u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -14 );
        }
                {
              uint64_t va[] = { 11814879313272076079u,7751630867623876719u,3283254022299871318u,13040651294719892351u }  ;
              uint64_t vb[] = { 6128846677797915938u,17840464375976882261u,6282874764732253364u,12892451889609190983u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -12 );
        }
                {
              uint64_t va[] = { 13674327918940713713u,2845224025248689127u,17103146914422845836u,12491881314271475653u }  ;
              uint64_t vb[] = { 15972229190526498277u,18112034750722485743u,17257274243824985025u,4764953030072647044u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  30 );
        }
                {
              uint64_t va[] = { 18120342129187467106u,9753476610488339454u,13575721824144335425u,14920767024976217347u }  ;
              uint64_t vb[] = { 17945046216146479135u,2533831551707093781u,4579877646471553359u,11472098830932056393u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  26 );
        }
                {
              uint64_t va[] = { 8645005125437181014u,1790089980848910552u,13069059305357658858u,6343166627355487142u }  ;
              uint64_t vb[] = { 13867750126591791149u,10773633957081645119u,11907208288301345718u,16808744385011298116u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 9534685354663597191u,7090592607922903774u,8130187926937854032u,5740711105760686332u }  ;
              uint64_t vb[] = { 11047938915461591752u,11568215744357432177u,6329783555630985017u,2902899359570805354u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  26 );
        }
                {
              uint64_t va[] = { 2356623829366914125u,15776300695562165180u,3328755472225181492u,1437153077145886027u }  ;
              uint64_t vb[] = { 2781376552192062778u,4058100772114783363u,6357544939396792581u,14065142130888537092u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  16 );
        }
                {
              uint64_t va[] = { 4864226140960223721u,14163233835160041290u,3492301929889900708u,490328144675494429u }  ;
              uint64_t vb[] = { 10010930966993168219u,5098359287480233117u,13655316623699959390u,2441328848382667241u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 16755476859930232205u,17197330873384242790u,7779590499473090003u,12511801534334719277u }  ;
              uint64_t vb[] = { 16816201024091620244u,10306911416287095277u,2986211059282155219u,1710778098649287669u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  14 );
        }
                {
              uint64_t va[] = { 8991840338369157365u,4743527857402854229u,2689124914758916514u,1573315985406522535u }  ;
              uint64_t vb[] = { 8594500690348089066u,2443560025766605798u,4527846380786373061u,2634342386134543781u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  28 );
        }
                {
              uint64_t va[] = { 313419615679578996u,14091986029129499258u,4906084607517482116u,15534477726357473662u }  ;
              uint64_t vb[] = { 10460418595007364946u,106513270812533108u,9263773134274729882u,2656024219489525132u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 10297899474108269126u,11429607204286485839u,18422728742630045389u,1851027355375734846u }  ;
              uint64_t vb[] = { 9222774902708377270u,10814152660637392543u,17831877635884108499u,3900928204498722814u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  24 );
        }
                {
              uint64_t va[] = { 15713450215836319322u,3261689969155682282u,9414829534409227966u,17093712354592355289u }  ;
              uint64_t vb[] = { 48366985044821553u,4481222890989363510u,10837476897109311336u,1676914513769166772u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 10590610615743985190u,5895585548328643184u,2207772480714848358u,10780999111291645560u }  ;
              uint64_t vb[] = { 9582917607570638389u,12771928634060825921u,17571576729463264868u,15264828584765602742u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 4189896460538193141u,244581026543387374u,7799733747929417043u,15843095476856850778u }  ;
              uint64_t vb[] = { 14475827798542019033u,14341002287761594896u,5356963810142035698u,16879250459884581815u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 12222719582286249035u,14101873082456828990u,11696295306981335365u,9792205245459447876u }  ;
              uint64_t vb[] = { 2141557210701585159u,10671420579183660580u,16644767407637517567u,5961752566369879668u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  34 );
        }
                {
              uint64_t va[] = { 688964272595065867u,359970898484531679u,11681404649968716440u,6783658121636059523u }  ;
              uint64_t vb[] = { 1635193044528780684u,11863640420126094280u,1130779507221165844u,15681029409643088821u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  8 );
        }
                {
              uint64_t va[] = { 12079209856812996759u,5774747623096770122u,14719988670268654049u,2303034158339517463u }  ;
              uint64_t vb[] = { 18301379289189344163u,2610210311811024264u,14284407410666354878u,6344086511095652591u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -26 );
        }
                {
              uint64_t va[] = { 15934318941156632082u,12892696647604246798u,13258379086060866362u,9322784125478260726u }  ;
              uint64_t vb[] = { 16673651760312609313u,5279610745622583172u,6451601535777504898u,7195650590844826595u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 5526134830403163737u,14460825600427327835u,3208840553004882077u,9552072020156167379u }  ;
              uint64_t vb[] = { 3172160486827244697u,15298525196812524791u,1507467840997724592u,10473332387405354173u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  16 );
        }
                {
              uint64_t va[] = { 1894053926100771149u,14076964913337454156u,684408350731893591u,15614544336920027805u }  ;
              uint64_t vb[] = { 15323996097341002650u,5306272098568118684u,9154864931479338045u,8848694625312257475u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 8379012994574932118u,823648419405336402u,15262268515898054553u,17708037173755287936u }  ;
              uint64_t vb[] = { 17091849398191995389u,13415346984103731580u,7508217079468618603u,2170592251068778836u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -14 );
        }
                {
              uint64_t va[] = { 11512304515811987893u,9362854730720804203u,6553368863878227854u,5798919576689333582u }  ;
              uint64_t vb[] = { 383014570419033078u,4603889493752904116u,6331714146985889304u,7010124717493374075u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 14957726587288862643u,15062314624160380400u,1078023492386323390u,16037432614028618576u }  ;
              uint64_t vb[] = { 2553094986650110915u,13954782299103361243u,6643773149242637585u,18347168541800034274u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 14266491197466318264u,16542078114655587745u,4718542359043532973u,17983233016530676761u }  ;
              uint64_t vb[] = { 6066998881675948554u,7404655507815241414u,17330083883622145508u,15732198930354084629u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -18 );
        }
                {
              uint64_t va[] = { 8162541939540690272u,6422228381704775903u,10942911184400853065u,4836049272167653951u }  ;
              uint64_t vb[] = { 10156672040348765157u,10534829552477493380u,9451170218072636727u,12312898068477678627u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 12726935858088489688u,2365390116179736506u,13483118975983236107u,15734514469026547392u }  ;
              uint64_t vb[] = { 245570970555104587u,17024137259274105515u,1860087783954940730u,1861762919310585179u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 11512826197733627395u,17490114092955900029u,17570429681864467308u,17858907062857067006u }  ;
              uint64_t vb[] = { 6039985020442576033u,16035478495651015485u,2127501989548446926u,9476275725955087950u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  24 );
        }
                {
              uint64_t va[] = { 8911017617183607202u,16111278479079487778u,12424252672764900507u,11881925895196292194u }  ;
              uint64_t vb[] = { 9923964613046598948u,5852745880758918690u,8913932198964067510u,17096386864158112449u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 9521458226930844497u,9624831537863406053u,17028157748502417046u,8262389642095793850u }  ;
              uint64_t vb[] = { 8561937969110642844u,17696023996015155175u,9185204177201207431u,11722355504193563138u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 11119025302546648135u,968784721299270964u,9442973057818427572u,788361383424783666u }  ;
              uint64_t vb[] = { 7200119099350796888u,14186977291613775782u,7853553820969024172u,13872167029391432808u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  10 );
        }
                {
              uint64_t va[] = { 6369586960445447614u,623015153342421987u,4009303807094392781u,4993684968031479951u }  ;
              uint64_t vb[] = { 16286482904570505527u,6469104178315502460u,3576571365577562828u,3878892653303924598u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  16 );
        }
                {
              uint64_t va[] = { 4916379289724183496u,2631239625055791917u,2851331458242276916u,1885638560067960333u }  ;
              uint64_t vb[] = { 12378286951977919921u,9257106658092594366u,3392277033036374522u,11032101420807632336u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -14 );
        }
                {
              uint64_t va[] = { 8374467907178237560u,4658442532330839350u,8090719111141703459u,7892122617168160698u }  ;
              uint64_t vb[] = { 18253370280201006258u,9215747423768776618u,8229572864329428906u,1316853242300348308u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  30 );
        }
                {
              uint64_t va[] = { 1652211731658960013u,9772054251930974453u,17686313136475510268u,12820751293378977329u }  ;
              uint64_t vb[] = { 571210816442553637u,14941038076532980387u,9172855300684248474u,14665603879903843831u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  20 );
        }
                {
              uint64_t va[] = { 16615925124778051608u,814443453627293098u,3295227470814889175u,2374191402225841380u }  ;
              uint64_t vb[] = { 11748675798448947567u,13913508493794744039u,3799206465942149323u,7664755266251056197u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  10 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint64_t*, const uint64_t*, size_t), dim=60") {
                {
              uint64_t va[] = { 1080529448920230110u }  ;
              uint64_t vb[] = { 12299494240976070u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 882955623165147605u }  ;
              uint64_t vb[] = { 984290731181992750u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 337075582801696364u }  ;
              uint64_t vb[] = { 1088719139603834290u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -12 );
        }
                {
              uint64_t va[] = { 1098945574201276011u }  ;
              uint64_t vb[] = { 305742158815236140u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 879774359585605577u }  ;
              uint64_t vb[] = { 611720993164063921u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 156927349769840748u }  ;
              uint64_t vb[] = { 162250468029832432u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 932435292593817170u }  ;
              uint64_t vb[] = { 21129616817192213u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -20 );
        }
                {
              uint64_t va[] = { 1074442451316746042u }  ;
              uint64_t vb[] = { 204423278749932560u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 11735988970068744u }  ;
              uint64_t vb[] = { 496053015413190741u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 998950669417692060u }  ;
              uint64_t vb[] = { 1000128348623294379u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 628126464660248888u }  ;
              uint64_t vb[] = { 1096962908918396070u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 989949361150282055u }  ;
              uint64_t vb[] = { 939466846288291150u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 708825066748088688u }  ;
              uint64_t vb[] = { 1063797038667945945u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 187673560805490631u }  ;
              uint64_t vb[] = { 709522953451115608u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 431331630192364023u }  ;
              uint64_t vb[] = { 174496765410408003u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 502376018888702708u }  ;
              uint64_t vb[] = { 369991444511623090u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 756259689220078474u }  ;
              uint64_t vb[] = { 922760596354849910u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 1053107406461255062u }  ;
              uint64_t vb[] = { 232952629013378165u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 339485978979404333u }  ;
              uint64_t vb[] = { 904820450631572254u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  10 );
        }
                {
              uint64_t va[] = { 554034544492036698u }  ;
              uint64_t vb[] = { 420126403095852744u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  10 );
        }
                {
              uint64_t va[] = { 747277095094497587u }  ;
              uint64_t vb[] = { 354799437075520583u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 176076703035169324u }  ;
              uint64_t vb[] = { 62363768586972162u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  8 );
        }
                {
              uint64_t va[] = { 660067243830858759u }  ;
              uint64_t vb[] = { 1062912203934996842u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 831401329532561142u }  ;
              uint64_t vb[] = { 640055017700051535u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 800880176114449197u }  ;
              uint64_t vb[] = { 1002480942926979941u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 413965064894604084u }  ;
              uint64_t vb[] = { 329092246522255316u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 83786853656474463u }  ;
              uint64_t vb[] = { 259332346377558011u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 866437604445606716u }  ;
              uint64_t vb[] = { 1047783259852786092u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  8 );
        }
                {
              uint64_t va[] = { 683567902953301150u }  ;
              uint64_t vb[] = { 44781216009262018u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 543513325382312039u }  ;
              uint64_t vb[] = { 1025537193685624629u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 55479258317533700u }  ;
              uint64_t vb[] = { 810717755888336166u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  8 );
        }
                {
              uint64_t va[] = { 1027965087346221056u }  ;
              uint64_t vb[] = { 477910859192989148u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 648427902099423429u }  ;
              uint64_t vb[] = { 418009672318934109u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  8 );
        }
                {
              uint64_t va[] = { 1058446746288331404u }  ;
              uint64_t vb[] = { 1092209417136447807u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 582498986684708707u }  ;
              uint64_t vb[] = { 734208992794027359u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 271457179883323245u }  ;
              uint64_t vb[] = { 572840133126426799u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 443009165558550646u }  ;
              uint64_t vb[] = { 344726324590850459u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 325961445354614250u }  ;
              uint64_t vb[] = { 544011429599024488u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 563573546985475043u }  ;
              uint64_t vb[] = { 288444958996086127u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 260900579933876229u }  ;
              uint64_t vb[] = { 821067029461018153u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 1042954348706523217u }  ;
              uint64_t vb[] = { 160651720509521275u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 671896677996669488u }  ;
              uint64_t vb[] = { 916164672871154495u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 920335711910033965u }  ;
              uint64_t vb[] = { 327540387148548423u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  8 );
        }
                {
              uint64_t va[] = { 897231624480557281u }  ;
              uint64_t vb[] = { 331234589770962474u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -14 );
        }
                {
              uint64_t va[] = { 97066235980586988u }  ;
              uint64_t vb[] = { 1075504776846783334u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 39842953628466150u }  ;
              uint64_t vb[] = { 1017088855606897748u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 994713030600773119u }  ;
              uint64_t vb[] = { 207629922698142248u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 385027256883750841u }  ;
              uint64_t vb[] = { 199733797002326477u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  8 );
        }
                {
              uint64_t va[] = { 179965332254434158u }  ;
              uint64_t vb[] = { 294102260940019137u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 1071640223177236277u }  ;
              uint64_t vb[] = { 947095710147646376u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -12 );
        }
                {
              uint64_t va[] = { 611451756466986464u }  ;
              uint64_t vb[] = { 41305609707967109u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 654201586536918322u }  ;
              uint64_t vb[] = { 712797472597740726u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 743725950732300784u }  ;
              uint64_t vb[] = { 911344659515575420u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 908271258172045395u }  ;
              uint64_t vb[] = { 807070105271848511u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 181262504967528157u }  ;
              uint64_t vb[] = { 25040691061224664u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  10 );
        }
                {
              uint64_t va[] = { 160898785592328759u }  ;
              uint64_t vb[] = { 991430001266217936u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 455093906732172364u }  ;
              uint64_t vb[] = { 591060415026115693u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 603868905448586902u }  ;
              uint64_t vb[] = { 588869541441179292u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  14 );
        }
                {
              uint64_t va[] = { 208093163699559010u }  ;
              uint64_t vb[] = { 655350515284336101u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 960268075335165229u }  ;
              uint64_t vb[] = { 794644015347320368u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 715393854876629551u }  ;
              uint64_t vb[] = { 213370645539087063u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 41499754962255167u }  ;
              uint64_t vb[] = { 932206359483579505u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 471327592553944793u }  ;
              uint64_t vb[] = { 684419278907233664u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 814547766069203955u }  ;
              uint64_t vb[] = { 231824150522678302u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 922257861061074846u }  ;
              uint64_t vb[] = { 262091114240589554u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 890263633925975494u }  ;
              uint64_t vb[] = { 592425569819778384u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 663705049698556279u }  ;
              uint64_t vb[] = { 113051964612179212u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  16 );
        }
                {
              uint64_t va[] = { 233457798808638647u }  ;
              uint64_t vb[] = { 68744595379261157u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 884968666359645154u }  ;
              uint64_t vb[] = { 931268674267677407u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 664453507411733715u }  ;
              uint64_t vb[] = { 1119329370242641570u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  10 );
        }
                {
              uint64_t va[] = { 558249002600604981u }  ;
              uint64_t vb[] = { 76117145706562501u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 967828474669011565u }  ;
              uint64_t vb[] = { 586503974750722208u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 158220783280585917u }  ;
              uint64_t vb[] = { 950877633581503710u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 898854136863244774u }  ;
              uint64_t vb[] = { 1031513013531221093u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 497341481619205785u }  ;
              uint64_t vb[] = { 779825298005636570u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 984396702981601864u }  ;
              uint64_t vb[] = { 1119890739836529537u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  14 );
        }
                {
              uint64_t va[] = { 451430353293310086u }  ;
              uint64_t vb[] = { 102916764858292365u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 756199535470326651u }  ;
              uint64_t vb[] = { 668138208620551103u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 113792433817479627u }  ;
              uint64_t vb[] = { 15921765132125882u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 206518987985438167u }  ;
              uint64_t vb[] = { 823285114307311885u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 954586670410518644u }  ;
              uint64_t vb[] = { 262184765337142172u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 1046733176408259066u }  ;
              uint64_t vb[] = { 976965318864348796u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 302399036042795066u }  ;
              uint64_t vb[] = { 827463739042313119u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 488078388657500093u }  ;
              uint64_t vb[] = { 409453468595947817u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 824289600715974083u }  ;
              uint64_t vb[] = { 711859586301242535u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 1023928928574468700u }  ;
              uint64_t vb[] = { 924081659100687658u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 1007087096702600835u }  ;
              uint64_t vb[] = { 593757087015260289u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 367163874985379733u }  ;
              uint64_t vb[] = { 997952375812762116u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 166059226136095312u }  ;
              uint64_t vb[] = { 483131809760683394u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -12 );
        }
                {
              uint64_t va[] = { 633660103416436894u }  ;
              uint64_t vb[] = { 606634850312532084u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  10 );
        }
                {
              uint64_t va[] = { 802534632418001614u }  ;
              uint64_t vb[] = { 1074172082436801682u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 1150701188477439397u }  ;
              uint64_t vb[] = { 492075191490296991u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 599129129493381320u }  ;
              uint64_t vb[] = { 1128703427211433618u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -14 );
        }
                {
              uint64_t va[] = { 661401331036264837u }  ;
              uint64_t vb[] = { 324515148065759990u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -12 );
        }
                {
              uint64_t va[] = { 659548099168691308u }  ;
              uint64_t vb[] = { 535643167459243897u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 1041232089646457198u }  ;
              uint64_t vb[] = { 263736738520615822u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 129367084056556198u }  ;
              uint64_t vb[] = { 725081103512597476u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 672709906009069400u }  ;
              uint64_t vb[] = { 213613727842448440u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 1043806218698584863u }  ;
              uint64_t vb[] = { 818038149851834869u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 106287130405367673u }  ;
              uint64_t vb[] = { 512326349967065860u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -14 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint64_t*, const uint64_t*, size_t), dim=59") {
                {
              uint64_t va[] = { 195415057196150606u }  ;
              uint64_t vb[] = { 228577449561712385u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  3 );
        }
                {
              uint64_t va[] = { 526390659373311651u }  ;
              uint64_t vb[] = { 493287877689398671u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  3 );
        }
                {
              uint64_t va[] = { 409010408875626402u }  ;
              uint64_t vb[] = { 449173166359348108u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  9 );
        }
                {
              uint64_t va[] = { 177332329992408085u }  ;
              uint64_t vb[] = { 168036738700539929u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  7 );
        }
                {
              uint64_t va[] = { 228676510398962944u }  ;
              uint64_t vb[] = { 385286492162193376u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -3 );
        }
                {
              uint64_t va[] = { 531060172781065605u }  ;
              uint64_t vb[] = { 362422999875171815u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  17 );
        }
                {
              uint64_t va[] = { 7298127395197578u }  ;
              uint64_t vb[] = { 229760022140211153u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -1 );
        }
                {
              uint64_t va[] = { 13734118347692249u }  ;
              uint64_t vb[] = { 410623991296823533u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  5 );
        }
                {
              uint64_t va[] = { 55755007920308212u }  ;
              uint64_t vb[] = { 248412894388786058u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -5 );
        }
                {
              uint64_t va[] = { 436507924090480779u }  ;
              uint64_t vb[] = { 519329467461887439u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  5 );
        }
                {
              uint64_t va[] = { 563898048132730517u }  ;
              uint64_t vb[] = { 493291533505736536u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  9 );
        }
                {
              uint64_t va[] = { 297142940453212087u }  ;
              uint64_t vb[] = { 396543101405090040u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -11 );
        }
                {
              uint64_t va[] = { 36687231051989178u }  ;
              uint64_t vb[] = { 308654143592310529u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -5 );
        }
                {
              uint64_t va[] = { 547748914523963454u }  ;
              uint64_t vb[] = { 375233292180012828u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  9 );
        }
                {
              uint64_t va[] = { 312573035638012596u }  ;
              uint64_t vb[] = { 446221364523077659u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  7 );
        }
                {
              uint64_t va[] = { 533232953982505698u }  ;
              uint64_t vb[] = { 172302487739733530u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  9 );
        }
                {
              uint64_t va[] = { 132872244717012941u }  ;
              uint64_t vb[] = { 78002441309314489u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  7 );
        }
                {
              uint64_t va[] = { 175919868434841249u }  ;
              uint64_t vb[] = { 149497031473296198u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 277237607333483830u }  ;
              uint64_t vb[] = { 74588133076261979u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  9 );
        }
                {
              uint64_t va[] = { 368483887810481468u }  ;
              uint64_t vb[] = { 103353920430222617u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  11 );
        }
                {
              uint64_t va[] = { 401747362595131558u }  ;
              uint64_t vb[] = { 516185841043213097u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -3 );
        }
                {
              uint64_t va[] = { 534974350634446134u }  ;
              uint64_t vb[] = { 116381814961070104u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -1 );
        }
                {
              uint64_t va[] = { 165104554027425670u }  ;
              uint64_t vb[] = { 531877088139772810u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  7 );
        }
                {
              uint64_t va[] = { 39423315396655324u }  ;
              uint64_t vb[] = { 138654896126418597u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -3 );
        }
                {
              uint64_t va[] = { 503923166442654582u }  ;
              uint64_t vb[] = { 300449483623053677u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  5 );
        }
                {
              uint64_t va[] = { 456904836082015084u }  ;
              uint64_t vb[] = { 112440633487470326u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  3 );
        }
                {
              uint64_t va[] = { 185671157871952936u }  ;
              uint64_t vb[] = { 163511809503616669u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -9 );
        }
                {
              uint64_t va[] = { 475285667537609153u }  ;
              uint64_t vb[] = { 172262907889033142u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -15 );
        }
                {
              uint64_t va[] = { 5785030948654393u }  ;
              uint64_t vb[] = { 527078482050957706u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  9 );
        }
                {
              uint64_t va[] = { 393292527787658390u }  ;
              uint64_t vb[] = { 377792204324967168u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -5 );
        }
                {
              uint64_t va[] = { 491569728597157604u }  ;
              uint64_t vb[] = { 75189962657834469u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 563357188519546007u }  ;
              uint64_t vb[] = { 72566969726108618u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -7 );
        }
                {
              uint64_t va[] = { 472349110697436985u }  ;
              uint64_t vb[] = { 292531154964504475u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  13 );
        }
                {
              uint64_t va[] = { 26581074986004577u }  ;
              uint64_t vb[] = { 526421184366220308u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -1 );
        }
                {
              uint64_t va[] = { 422343918658247809u }  ;
              uint64_t vb[] = { 123940355059698580u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  17 );
        }
                {
              uint64_t va[] = { 451351964583488497u }  ;
              uint64_t vb[] = { 399183140572334978u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -5 );
        }
                {
              uint64_t va[] = { 440253157775716232u }  ;
              uint64_t vb[] = { 247271555751176603u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 372678976523259301u }  ;
              uint64_t vb[] = { 470168424600718318u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  7 );
        }
                {
              uint64_t va[] = { 34335586713849865u }  ;
              uint64_t vb[] = { 295601618488752168u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  9 );
        }
                {
              uint64_t va[] = { 478436260134595593u }  ;
              uint64_t vb[] = { 156446519304002262u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -3 );
        }
                {
              uint64_t va[] = { 499530443645947704u }  ;
              uint64_t vb[] = { 292369148037417215u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  7 );
        }
                {
              uint64_t va[] = { 324502610738567213u }  ;
              uint64_t vb[] = { 333776914215782382u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 324358598785795547u }  ;
              uint64_t vb[] = { 490219651103050758u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  5 );
        }
                {
              uint64_t va[] = { 216749681236314264u }  ;
              uint64_t vb[] = { 389848756321944692u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -9 );
        }
                {
              uint64_t va[] = { 292046538205496291u }  ;
              uint64_t vb[] = { 569513359302677294u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -7 );
        }
                {
              uint64_t va[] = { 256035289726372360u }  ;
              uint64_t vb[] = { 86805923594607507u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -11 );
        }
                {
              uint64_t va[] = { 337756215235833769u }  ;
              uint64_t vb[] = { 233342769784001608u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -17 );
        }
                {
              uint64_t va[] = { 267286714909975433u }  ;
              uint64_t vb[] = { 279811156282459977u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  19 );
        }
                {
              uint64_t va[] = { 395138068793983298u }  ;
              uint64_t vb[] = { 418676689931179388u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 18871519576665368u }  ;
              uint64_t vb[] = { 200007838315196836u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -7 );
        }
                {
              uint64_t va[] = { 389244835209960562u }  ;
              uint64_t vb[] = { 412618073208252259u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  3 );
        }
                {
              uint64_t va[] = { 180355081828773136u }  ;
              uint64_t vb[] = { 146216856587525985u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  7 );
        }
                {
              uint64_t va[] = { 500973220964468294u }  ;
              uint64_t vb[] = { 153271268427674477u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -5 );
        }
                {
              uint64_t va[] = { 369775214871234393u }  ;
              uint64_t vb[] = { 241388047498516938u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  3 );
        }
                {
              uint64_t va[] = { 54870584559533395u }  ;
              uint64_t vb[] = { 545913251105135474u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  11 );
        }
                {
              uint64_t va[] = { 185539801755331029u }  ;
              uint64_t vb[] = { 113465244013074659u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  3 );
        }
                {
              uint64_t va[] = { 566468265475379997u }  ;
              uint64_t vb[] = { 472692682254841668u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  5 );
        }
                {
              uint64_t va[] = { 9795479316119629u }  ;
              uint64_t vb[] = { 311335332329936845u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  13 );
        }
                {
              uint64_t va[] = { 573780827692606080u }  ;
              uint64_t vb[] = { 317855368647163889u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -9 );
        }
                {
              uint64_t va[] = { 432662960638195406u }  ;
              uint64_t vb[] = { 150900487164694962u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 144988495434487446u }  ;
              uint64_t vb[] = { 439350339992934569u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -9 );
        }
                {
              uint64_t va[] = { 256239339935528202u }  ;
              uint64_t vb[] = { 536454609451119057u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 451431517420693270u }  ;
              uint64_t vb[] = { 21384552531785313u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  3 );
        }
                {
              uint64_t va[] = { 350147564345176984u }  ;
              uint64_t vb[] = { 273990431520527077u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  13 );
        }
                {
              uint64_t va[] = { 72233517517742225u }  ;
              uint64_t vb[] = { 309075603342528033u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -7 );
        }
                {
              uint64_t va[] = { 11603545288589186u }  ;
              uint64_t vb[] = { 476372442884237496u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -9 );
        }
                {
              uint64_t va[] = { 412902514154735979u }  ;
              uint64_t vb[] = { 295660173221800589u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  7 );
        }
                {
              uint64_t va[] = { 173097683533109100u }  ;
              uint64_t vb[] = { 443739238884469330u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -1 );
        }
                {
              uint64_t va[] = { 428243591035032239u }  ;
              uint64_t vb[] = { 454579645985115990u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -9 );
        }
                {
              uint64_t va[] = { 80669244191857691u }  ;
              uint64_t vb[] = { 226718178452316577u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -17 );
        }
                {
              uint64_t va[] = { 307426326046485064u }  ;
              uint64_t vb[] = { 439208055169113325u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -3 );
        }
                {
              uint64_t va[] = { 92207247113148260u }  ;
              uint64_t vb[] = { 199606225419882940u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  3 );
        }
                {
              uint64_t va[] = { 227611690125473683u }  ;
              uint64_t vb[] = { 113722014376429458u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  7 );
        }
                {
              uint64_t va[] = { 20652046092074922u }  ;
              uint64_t vb[] = { 156380861323067138u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -3 );
        }
                {
              uint64_t va[] = { 373896421219300500u }  ;
              uint64_t vb[] = { 269835409576670760u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -5 );
        }
                {
              uint64_t va[] = { 569946043060906421u }  ;
              uint64_t vb[] = { 351786189867010173u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  19 );
        }
                {
              uint64_t va[] = { 542951237318432463u }  ;
              uint64_t vb[] = { 406152304090934021u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  9 );
        }
                {
              uint64_t va[] = { 194446673601522391u }  ;
              uint64_t vb[] = { 22205145651352853u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -9 );
        }
                {
              uint64_t va[] = { 266799002029594611u }  ;
              uint64_t vb[] = { 119919789313698334u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -9 );
        }
                {
              uint64_t va[] = { 261028189108574889u }  ;
              uint64_t vb[] = { 47966364971952524u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -11 );
        }
                {
              uint64_t va[] = { 51581176761824985u }  ;
              uint64_t vb[] = { 357118209585974735u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 180897600318395820u }  ;
              uint64_t vb[] = { 348133623511450933u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -3 );
        }
                {
              uint64_t va[] = { 105614621470415771u }  ;
              uint64_t vb[] = { 130188858669525909u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  9 );
        }
                {
              uint64_t va[] = { 139453190733345137u }  ;
              uint64_t vb[] = { 244546487996514554u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  5 );
        }
                {
              uint64_t va[] = { 76290067590843236u }  ;
              uint64_t vb[] = { 10830714494100506u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -1 );
        }
                {
              uint64_t va[] = { 2215656967062067u }  ;
              uint64_t vb[] = { 172281042812129317u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  7 );
        }
                {
              uint64_t va[] = { 338830469198854474u }  ;
              uint64_t vb[] = { 215876495178548768u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -3 );
        }
                {
              uint64_t va[] = { 570959135108438268u }  ;
              uint64_t vb[] = { 277051123905307830u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  3 );
        }
                {
              uint64_t va[] = { 495592715383825523u }  ;
              uint64_t vb[] = { 200848797300400460u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  5 );
        }
                {
              uint64_t va[] = { 541339549413030924u }  ;
              uint64_t vb[] = { 437775946565056295u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  3 );
        }
                {
              uint64_t va[] = { 211961070805712391u }  ;
              uint64_t vb[] = { 360506562949286761u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -23 );
        }
                {
              uint64_t va[] = { 524049511448750575u }  ;
              uint64_t vb[] = { 181947431643997761u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -1 );
        }
                {
              uint64_t va[] = { 262041608678061751u }  ;
              uint64_t vb[] = { 36634916153841409u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  5 );
        }
                {
              uint64_t va[] = { 231497877514078480u }  ;
              uint64_t vb[] = { 316583608397628952u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  7 );
        }
                {
              uint64_t va[] = { 367829443150442678u }  ;
              uint64_t vb[] = { 180987160159992965u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -5 );
        }
                {
              uint64_t va[] = { 199564978450913815u }  ;
              uint64_t vb[] = { 283646894426016067u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  9 );
        }
                {
              uint64_t va[] = { 561451289146453042u }  ;
              uint64_t vb[] = { 272104469981879129u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -3 );
        }
                {
              uint64_t va[] = { 276128261737220225u }  ;
              uint64_t vb[] = { 220313578091739348u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  3 );
        }
                {
              uint64_t va[] = { 181181445222227075u }  ;
              uint64_t vb[] = { 154149279442939156u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  9 );
        }
                {
              uint64_t va[] = { 560641757843421670u }  ;
              uint64_t vb[] = { 464023579306839127u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -7 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint64_t*, const uint64_t*, size_t), dim=100") {
                {
              uint64_t va[] = { 16832606093628846901u,15413133571u }  ;
              uint64_t vb[] = { 15150663611773064589u,61168090917u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 2456658357776263673u,67417986091u }  ;
              uint64_t vb[] = { 10650907461075013936u,67247835565u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  22 );
        }
                {
              uint64_t va[] = { 1669530526436844022u,20309084668u }  ;
              uint64_t vb[] = { 8436257192685597231u,55383545309u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  8 );
        }
                {
              uint64_t va[] = { 15864372237423922765u,31258905189u }  ;
              uint64_t vb[] = { 17101757923721976931u,65101042774u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 4653952568467155219u,6068754321u }  ;
              uint64_t vb[] = { 7971817987948212079u,36616278256u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 14490105501554056491u,50144915254u }  ;
              uint64_t vb[] = { 14230022901428285916u,53595816004u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 7698945454229327098u,61565962849u }  ;
              uint64_t vb[] = { 17106721703903665784u,60595576707u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 8320747614054407102u,26340089940u }  ;
              uint64_t vb[] = { 15435529275179494612u,49076308121u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -12 );
        }
                {
              uint64_t va[] = { 124363756610654970u,17147547200u }  ;
              uint64_t vb[] = { 17216253583349674740u,10098939933u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 2224233326410899490u,68455099760u }  ;
              uint64_t vb[] = { 3152666195579330041u,55701406978u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 2303282650720677592u,36729818666u }  ;
              uint64_t vb[] = { 8504785627832413082u,58518599542u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 9560304457684114165u,36391850878u }  ;
              uint64_t vb[] = { 8466722603691901039u,66856057259u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 473311515879617670u,23424255401u }  ;
              uint64_t vb[] = { 3664654808745443752u,1811746291u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 12328842684438534285u,61580241782u }  ;
              uint64_t vb[] = { 11325615616075753109u,56824672571u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 4857204792014313237u,23861702110u }  ;
              uint64_t vb[] = { 8059300844425325235u,22858487269u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 1693226349031534411u,12362652088u }  ;
              uint64_t vb[] = { 17559661535979918678u,45710666146u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 1902542310201916038u,15950758917u }  ;
              uint64_t vb[] = { 13690448751184026208u,48082174105u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 16705522351156618966u,3504474662u }  ;
              uint64_t vb[] = { 2951549593896330152u,24660644453u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  8 );
        }
                {
              uint64_t va[] = { 17700426827223127658u,46420282638u }  ;
              uint64_t vb[] = { 7153469556892688013u,55192085990u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 17545093396217867437u,15235827673u }  ;
              uint64_t vb[] = { 226735603785622369u,67589061126u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -16 );
        }
                {
              uint64_t va[] = { 14090719566101939888u,41506816634u }  ;
              uint64_t vb[] = { 9814016391154874662u,22010115142u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 1037187676745678363u,26214321845u }  ;
              uint64_t vb[] = { 9324692654077678749u,22511185333u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  22 );
        }
                {
              uint64_t va[] = { 17612290165288956834u,19482783767u }  ;
              uint64_t vb[] = { 1158466485229478509u,6736565958u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 1668393643337555739u,17103661493u }  ;
              uint64_t vb[] = { 1917626046285641878u,25685595308u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 3588584697630578692u,9203940602u }  ;
              uint64_t vb[] = { 11498522998039488942u,8812459703u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 17424042571748174314u,25635022726u }  ;
              uint64_t vb[] = { 6204119745079802079u,11730858418u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 8613340118985620756u,52124994712u }  ;
              uint64_t vb[] = { 1647236566287661315u,53134764050u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  10 );
        }
                {
              uint64_t va[] = { 14475844882860896897u,65385418689u }  ;
              uint64_t vb[] = { 16135934628249878751u,44537868854u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -16 );
        }
                {
              uint64_t va[] = { 8259224317253028032u,7312447008u }  ;
              uint64_t vb[] = { 2223438953046726571u,15972969042u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 8890088413900536751u,62890583490u }  ;
              uint64_t vb[] = { 2485213326822182693u,2864719591u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  22 );
        }
                {
              uint64_t va[] = { 5756417980334453170u,12229301925u }  ;
              uint64_t vb[] = { 6738795161349223476u,5897998092u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 8628610424086554903u,36284903459u }  ;
              uint64_t vb[] = { 8452342085762572641u,52787861272u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 5745202974102693658u,34750622063u }  ;
              uint64_t vb[] = { 9475775138611528578u,10319790338u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 8756620149750486118u,1459534525u }  ;
              uint64_t vb[] = { 10119457283642276732u,31318957469u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -16 );
        }
                {
              uint64_t va[] = { 17664017763606297434u,41523203839u }  ;
              uint64_t vb[] = { 960567406459632709u,25592849034u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -12 );
        }
                {
              uint64_t va[] = { 495887700027945629u,6129428554u }  ;
              uint64_t vb[] = { 13668789194913945724u,45867678850u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 520118383994390831u,31196704344u }  ;
              uint64_t vb[] = { 2161587826598961792u,60970088763u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 5729235376081071547u,12290855357u }  ;
              uint64_t vb[] = { 15952594620046441040u,28858472955u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  12 );
        }
                {
              uint64_t va[] = { 1647853080623065997u,14228340799u }  ;
              uint64_t vb[] = { 6892551419317848342u,60536164537u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  14 );
        }
                {
              uint64_t va[] = { 4968012430839095269u,2559492078u }  ;
              uint64_t vb[] = { 2258948819816743149u,62170581449u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  18 );
        }
                {
              uint64_t va[] = { 16088032752630478214u,33925881420u }  ;
              uint64_t vb[] = { 7650671485109390133u,35093309480u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 17424237308072308623u,48576414668u }  ;
              uint64_t vb[] = { 17361474548249435981u,61439414852u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  24 );
        }
                {
              uint64_t va[] = { 15851844111528667910u,36321685678u }  ;
              uint64_t vb[] = { 6572285915764843248u,44814288643u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 4115218269077350620u,57946507134u }  ;
              uint64_t vb[] = { 8401130747250936106u,35522837028u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 3712626671936055233u,44627280512u }  ;
              uint64_t vb[] = { 9489162459998715329u,580084043u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  20 );
        }
                {
              uint64_t va[] = { 478215310052178338u,4341008459u }  ;
              uint64_t vb[] = { 16887777211172164050u,53879307072u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 3594094093732772935u,10175069438u }  ;
              uint64_t vb[] = { 11683001257930927274u,4862941270u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 7807480876778851409u,7576528373u }  ;
              uint64_t vb[] = { 2063699110642140639u,61750051215u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 287305383895815450u,46346338450u }  ;
              uint64_t vb[] = { 16224400337973995412u,65175605331u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  14 );
        }
                {
              uint64_t va[] = { 7827561261722080524u,15644376261u }  ;
              uint64_t vb[] = { 7670422074778665776u,7022203599u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  12 );
        }
                {
              uint64_t va[] = { 3650240947161007606u,3251360296u }  ;
              uint64_t vb[] = { 17885743956944407778u,9385257906u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 12090986770887066679u,68198162996u }  ;
              uint64_t vb[] = { 15787057194226334362u,37839649264u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 14543396229159867607u,53915236164u }  ;
              uint64_t vb[] = { 8455317042044485266u,33188776660u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 17239509053462294147u,46005416962u }  ;
              uint64_t vb[] = { 9634737648210493724u,55022575433u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -16 );
        }
                {
              uint64_t va[] = { 4039764535756347848u,64808080092u }  ;
              uint64_t vb[] = { 18373112014307992854u,58435219348u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 8620855777176597873u,36168314254u }  ;
              uint64_t vb[] = { 18344568234803906228u,24084083686u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 17589072608160965645u,44025387097u }  ;
              uint64_t vb[] = { 15507699819925513906u,5741574943u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 4697514660044865493u,5528387384u }  ;
              uint64_t vb[] = { 9652786097608180395u,9665586491u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 5142906136551286300u,61202896537u }  ;
              uint64_t vb[] = { 7441831543421352575u,20793111187u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 14823731237378078350u,8543087503u }  ;
              uint64_t vb[] = { 2804908201879499992u,5809891898u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 18145253857380379571u,42922404453u }  ;
              uint64_t vb[] = { 995662254872395481u,44986447522u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 17784939278885567760u,62686007140u }  ;
              uint64_t vb[] = { 14481680830542935106u,64496689175u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 495584901028678512u,44197680817u }  ;
              uint64_t vb[] = { 7378956660480390784u,5182510770u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 11238058112655785278u,25073148234u }  ;
              uint64_t vb[] = { 4752554418317502226u,47230570856u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 5383033274614127104u,51157002519u }  ;
              uint64_t vb[] = { 6698689822633942259u,22561195635u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 16147951121700226546u,8583311920u }  ;
              uint64_t vb[] = { 6021870950303047509u,51726770265u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 9310126135588821887u,42537881037u }  ;
              uint64_t vb[] = { 9561512351523695476u,6804324296u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  8 );
        }
                {
              uint64_t va[] = { 9713097596540498615u,66502334132u }  ;
              uint64_t vb[] = { 9619149171247502108u,12961217442u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -16 );
        }
                {
              uint64_t va[] = { 11102967875211046958u,65224749333u }  ;
              uint64_t vb[] = { 4964397591167894972u,32345323808u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 4666832287468630101u,13820874702u }  ;
              uint64_t vb[] = { 6724218709662602885u,57449814903u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 4902328018876697582u,36385986768u }  ;
              uint64_t vb[] = { 16749935839622298363u,66676821326u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -16 );
        }
                {
              uint64_t va[] = { 523119900457931896u,1270295838u }  ;
              uint64_t vb[] = { 9771335105984242809u,9134609298u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  18 );
        }
                {
              uint64_t va[] = { 2963572074896699319u,60845360400u }  ;
              uint64_t vb[] = { 10528251907330632221u,14950737090u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 7495935673176796056u,26706422317u }  ;
              uint64_t vb[] = { 8529244046049100696u,58426676281u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 3767952100247674785u,58404819981u }  ;
              uint64_t vb[] = { 12145764587673967037u,5657518705u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  20 );
        }
                {
              uint64_t va[] = { 14697284064736442667u,61082726104u }  ;
              uint64_t vb[] = { 16051775598462802427u,59379605923u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  8 );
        }
                {
              uint64_t va[] = { 3685259145225097638u,39446291158u }  ;
              uint64_t vb[] = { 14690579833442360945u,37203642750u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -12 );
        }
                {
              uint64_t va[] = { 6307959146017207133u,22591464683u }  ;
              uint64_t vb[] = { 11313813594819164279u,10770151756u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 8267896741390381292u,20616784949u }  ;
              uint64_t vb[] = { 11604481927240833211u,15147155932u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 14474708858461592085u,45689864701u }  ;
              uint64_t vb[] = { 10779060955393313628u,17211036342u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 11038016749950060652u,61156170974u }  ;
              uint64_t vb[] = { 16537674550417911993u,29545915007u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 16433769750318210293u,7715288013u }  ;
              uint64_t vb[] = { 8604360157073383099u,3675678427u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  8 );
        }
                {
              uint64_t va[] = { 2136034428798596739u,22104791823u }  ;
              uint64_t vb[] = { 1004814239640396828u,18330328232u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 591447429075671362u,48540976665u }  ;
              uint64_t vb[] = { 15383179216907049627u,11496342536u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 18168766482146673054u,30231325649u }  ;
              uint64_t vb[] = { 16152905022571655325u,39622237813u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  16 );
        }
                {
              uint64_t va[] = { 13683120633118802868u,25366558899u }  ;
              uint64_t vb[] = { 3284108958897415326u,27560240837u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  8 );
        }
                {
              uint64_t va[] = { 3029865819215099956u,8199956145u }  ;
              uint64_t vb[] = { 9528341794513571438u,55991839396u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 14971576893772104478u,22115177811u }  ;
              uint64_t vb[] = { 4138565415745863898u,42434532347u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 16873725106299375470u,624287967u }  ;
              uint64_t vb[] = { 15730057215795845724u,29765824818u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  10 );
        }
                {
              uint64_t va[] = { 280668324604436170u,25715575990u }  ;
              uint64_t vb[] = { 18360321282387100196u,19928002899u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -18 );
        }
                {
              uint64_t va[] = { 6354858010589439822u,24537810734u }  ;
              uint64_t vb[] = { 12109941909106685487u,26994629856u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 14147017741070096525u,6524352729u }  ;
              uint64_t vb[] = { 12261711110688229761u,61660642922u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 7070067048886578905u,65763440889u }  ;
              uint64_t vb[] = { 15485150370336707662u,51811954546u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -22 );
        }
                {
              uint64_t va[] = { 15062854200101917871u,65422021680u }  ;
              uint64_t vb[] = { 3044773348926016069u,56674132848u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 18410435517525027181u,52090008019u }  ;
              uint64_t vb[] = { 350722231815629864u,55791069788u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 18221904192174376021u,64075893624u }  ;
              uint64_t vb[] = { 10609699675579209618u,26001413741u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -14 );
        }
                {
              uint64_t va[] = { 2871443532537062934u,12934764411u }  ;
              uint64_t vb[] = { 11634482996391044540u,13687371126u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  26 );
        }
                {
              uint64_t va[] = { 17128952429554974115u,55315746094u }  ;
              uint64_t vb[] = { 18377652730906350085u,25534087326u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  14 );
        }
                {
              uint64_t va[] = { 17472114293923923635u,31212653232u }  ;
              uint64_t vb[] = { 13265359445059018683u,28445695742u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  16 );
        }
                {
              uint64_t va[] = { 1663922864119963008u,59286392542u }  ;
              uint64_t vb[] = { 9788431935051030358u,325199502u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint64_t*, const uint64_t*, size_t), dim=119") {
                {
              uint64_t va[] = { 17773813333626147531u,17510638882306530u }  ;
              uint64_t vb[] = { 15228234900279049997u,686102440650868u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -21 );
        }
                {
              uint64_t va[] = { 6156667683368654784u,4637336673216451u }  ;
              uint64_t vb[] = { 5135411972892674833u,5669980095160082u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  15 );
        }
                {
              uint64_t va[] = { 414665317075195976u,30216015998771192u }  ;
              uint64_t vb[] = { 14012560164681970462u,20362000412433388u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -9 );
        }
                {
              uint64_t va[] = { 2802362916374928309u,7559827749750140u }  ;
              uint64_t vb[] = { 10766189018402942227u,16410818255874320u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 5329095425874500197u,7041026805798164u }  ;
              uint64_t vb[] = { 8200616449949018247u,31950493695359042u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  15 );
        }
                {
              uint64_t va[] = { 6366504417118962425u,17074533953576234u }  ;
              uint64_t vb[] = { 2522182224763059503u,1985589624691552u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -15 );
        }
                {
              uint64_t va[] = { 420733400070231045u,16556679635156090u }  ;
              uint64_t vb[] = { 4475226540536845379u,24321705018199284u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  9 );
        }
                {
              uint64_t va[] = { 14901970104974686152u,35202731891504139u }  ;
              uint64_t vb[] = { 3975070021625941721u,24056874583993588u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -1 );
        }
                {
              uint64_t va[] = { 3169071128703981977u,12483901295881301u }  ;
              uint64_t vb[] = { 6206672619805292832u,1939402196216794u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -17 );
        }
                {
              uint64_t va[] = { 1765446321990576825u,20848354210271504u }  ;
              uint64_t vb[] = { 6375774568983572759u,1448474307572960u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -3 );
        }
                {
              uint64_t va[] = { 7636802149912244877u,946618460663424u }  ;
              uint64_t vb[] = { 9642402806036494735u,7195347499662534u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -3 );
        }
                {
              uint64_t va[] = { 14324935870105947424u,32229247070906864u }  ;
              uint64_t vb[] = { 16394307782961274579u,8612342068419827u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  3 );
        }
                {
              uint64_t va[] = { 3630032488221241974u,29794374802944045u }  ;
              uint64_t vb[] = { 15711933796507835038u,10717264071089888u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  3 );
        }
                {
              uint64_t va[] = { 14422441945040175644u,14277802446366193u }  ;
              uint64_t vb[] = { 16657277816818622003u,6254128190798263u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  9 );
        }
                {
              uint64_t va[] = { 7370871755851168725u,21451700159461851u }  ;
              uint64_t vb[] = { 873315314490515810u,6879781122598826u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -11 );
        }
                {
              uint64_t va[] = { 6185705505981501499u,5292420578584813u }  ;
              uint64_t vb[] = { 9587059790822784711u,2136078623822225u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -13 );
        }
                {
              uint64_t va[] = { 4260228097034625497u,30781967248987623u }  ;
              uint64_t vb[] = { 4844778488463239254u,13041185563300901u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  11 );
        }
                {
              uint64_t va[] = { 7288714977133312763u,20443677180902451u }  ;
              uint64_t vb[] = { 3981147033412151064u,26716310563068145u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  9 );
        }
                {
              uint64_t va[] = { 9031838072483486296u,19373979517950618u }  ;
              uint64_t vb[] = { 10679322834564863517u,1123447709650850u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  7 );
        }
                {
              uint64_t va[] = { 16403402572598058322u,9289737032125073u }  ;
              uint64_t vb[] = { 10215007205116312652u,9587201290132753u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  9 );
        }
                {
              uint64_t va[] = { 13074408894989304989u,3248155529374950u }  ;
              uint64_t vb[] = { 16858821762149088437u,19999822726594807u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  13 );
        }
                {
              uint64_t va[] = { 9089740093358423062u,18036026314832889u }  ;
              uint64_t vb[] = { 15582563034144162146u,13984884936850533u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -3 );
        }
                {
              uint64_t va[] = { 3103680034912289804u,29979701568035132u }  ;
              uint64_t vb[] = { 2979487026416842348u,1740305399989346u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 8612890390313117687u,5761898443297275u }  ;
              uint64_t vb[] = { 2088043359500571623u,32302099682273658u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 4804656211560882672u,35396606572038929u }  ;
              uint64_t vb[] = { 11235216889112622300u,19830310961083701u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -1 );
        }
                {
              uint64_t va[] = { 1012581787706368567u,18192248037957820u }  ;
              uint64_t vb[] = { 298034406405309843u,34845067516325737u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -21 );
        }
                {
              uint64_t va[] = { 5378482088693300367u,15425487877756861u }  ;
              uint64_t vb[] = { 1240858826046059607u,5170718043610306u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -15 );
        }
                {
              uint64_t va[] = { 2938375611284519595u,32678589120510041u }  ;
              uint64_t vb[] = { 12157791126590205704u,8614678903980812u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  7 );
        }
                {
              uint64_t va[] = { 8027501837457781327u,23798234494463303u }  ;
              uint64_t vb[] = { 331223119153565932u,12111600105532703u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -19 );
        }
                {
              uint64_t va[] = { 17178765007914030645u,13756914859165108u }  ;
              uint64_t vb[] = { 3711849549293946227u,10450607886810265u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -11 );
        }
                {
              uint64_t va[] = { 12730004191876971155u,16389936008061837u }  ;
              uint64_t vb[] = { 4622410822724110443u,25137373783542851u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -1 );
        }
                {
              uint64_t va[] = { 12217085874221321720u,7173791476706517u }  ;
              uint64_t vb[] = { 4914227240637529343u,18809587757928480u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -15 );
        }
                {
              uint64_t va[] = { 14052817908406269791u,33407677799684091u }  ;
              uint64_t vb[] = { 17698940240760431302u,33658917777635983u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 1868118898299712560u,21345259457526066u }  ;
              uint64_t vb[] = { 2389346540731842042u,5386694463776190u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -3 );
        }
                {
              uint64_t va[] = { 16719876921404758432u,10628567087826061u }  ;
              uint64_t vb[] = { 3812379471076575630u,11588183836706148u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  13 );
        }
                {
              uint64_t va[] = { 17403491000368286095u,694818182573648u }  ;
              uint64_t vb[] = { 17613574657759112136u,30332236342293259u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 6897996269353405495u,20892653828293210u }  ;
              uint64_t vb[] = { 5151765584311274868u,9557403019412663u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  13 );
        }
                {
              uint64_t va[] = { 3412464715648239648u,9258970116630973u }  ;
              uint64_t vb[] = { 7377082242216428989u,79558718296000u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 4359272839048979955u,30548847934316236u }  ;
              uint64_t vb[] = { 10606299664438908230u,26862401264212333u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -1 );
        }
                {
              uint64_t va[] = { 8694336700657731320u,19121086990034847u }  ;
              uint64_t vb[] = { 3140896739223008038u,32072741158665313u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -23 );
        }
                {
              uint64_t va[] = { 13906494089404000420u,14097460822368716u }  ;
              uint64_t vb[] = { 151823364275629518u,244549394336205u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -3 );
        }
                {
              uint64_t va[] = { 4125802361849514217u,35777701384828367u }  ;
              uint64_t vb[] = { 4861424926661235396u,2495371015848145u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -11 );
        }
                {
              uint64_t va[] = { 8744330720585885543u,9624206903797190u }  ;
              uint64_t vb[] = { 414509715218579990u,18644776397497315u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  19 );
        }
                {
              uint64_t va[] = { 17442357565366023199u,20862900229935242u }  ;
              uint64_t vb[] = { 8401518202108281645u,24434999366765220u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  7 );
        }
                {
              uint64_t va[] = { 16730920290807297924u,12088324943369336u }  ;
              uint64_t vb[] = { 10259392718409010232u,22791257273934252u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  7 );
        }
                {
              uint64_t va[] = { 16423014202191396388u,4217999014645106u }  ;
              uint64_t vb[] = { 15803371184810774053u,2885029799713289u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  5 );
        }
                {
              uint64_t va[] = { 7147023411140985636u,22452282957571653u }  ;
              uint64_t vb[] = { 702172279397027631u,18384928241050782u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  11 );
        }
                {
              uint64_t va[] = { 8042068505515700490u,34633987813959657u }  ;
              uint64_t vb[] = { 18216793387783596551u,34516686817437215u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  3 );
        }
                {
              uint64_t va[] = { 11031539427156646228u,11389093509511562u }  ;
              uint64_t vb[] = { 11359222560191696573u,14985994071128948u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -15 );
        }
                {
              uint64_t va[] = { 16821585490721233231u,11772035066368467u }  ;
              uint64_t vb[] = { 4281063272296757664u,13498607132615199u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 2809319397362016969u,14709996141588095u }  ;
              uint64_t vb[] = { 6538581978237979542u,6100011364131893u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  13 );
        }
                {
              uint64_t va[] = { 16922503294240874430u,32055738000347135u }  ;
              uint64_t vb[] = { 14699202969385058456u,19048441193637249u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  7 );
        }
                {
              uint64_t va[] = { 12794291432694080059u,25950321316528372u }  ;
              uint64_t vb[] = { 4554146361341731611u,12025558903783855u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  11 );
        }
                {
              uint64_t va[] = { 13219105709282156046u,10726337083071377u }  ;
              uint64_t vb[] = { 10428032342190930481u,19094686742239471u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -1 );
        }
                {
              uint64_t va[] = { 11210088949245495716u,17839890215286021u }  ;
              uint64_t vb[] = { 1140945202846819374u,735719596934476u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -1 );
        }
                {
              uint64_t va[] = { 13943905637059151629u,15141145802423351u }  ;
              uint64_t vb[] = { 5277728294634994082u,16462907634622280u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -17 );
        }
                {
              uint64_t va[] = { 3236964541542229686u,23728564724527556u }  ;
              uint64_t vb[] = { 1897692993434581954u,15406616841041362u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -9 );
        }
                {
              uint64_t va[] = { 13304538503936782353u,31808296994850292u }  ;
              uint64_t vb[] = { 7289748670105427606u,31151776809741799u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  5 );
        }
                {
              uint64_t va[] = { 7417703561930154438u,9367092923501522u }  ;
              uint64_t vb[] = { 7180574812753190245u,29823683626822480u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  13 );
        }
                {
              uint64_t va[] = { 18176487475799211650u,27197993197996248u }  ;
              uint64_t vb[] = { 15897370323956229643u,9231692881164052u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  11 );
        }
                {
              uint64_t va[] = { 17800101143196372325u,29814089349298400u }  ;
              uint64_t vb[] = { 5478846329316702258u,6391066650111250u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -19 );
        }
                {
              uint64_t va[] = { 14446934757845533702u,2243109287580430u }  ;
              uint64_t vb[] = { 6024815862441938677u,26200443559530587u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -3 );
        }
                {
              uint64_t va[] = { 7831208430304554005u,15269389898319734u }  ;
              uint64_t vb[] = { 14846158504217591583u,15791425365503335u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  13 );
        }
                {
              uint64_t va[] = { 16886437943103755996u,19391485522175900u }  ;
              uint64_t vb[] = { 4622441240951873537u,17961948884314284u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 18374546301264891875u,14106298845492915u }  ;
              uint64_t vb[] = { 13024854290758105349u,13087168336255022u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 13327294420801765138u,18093454821775434u }  ;
              uint64_t vb[] = { 116261122423278387u,3438861176900234u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  5 );
        }
                {
              uint64_t va[] = { 13904423398381069631u,1853323374871497u }  ;
              uint64_t vb[] = { 3790519181043344290u,33882842438915273u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -13 );
        }
                {
              uint64_t va[] = { 3610511757940772860u,6269561575047928u }  ;
              uint64_t vb[] = { 5279498411683898816u,4717829341933187u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -5 );
        }
                {
              uint64_t va[] = { 9743877291906854958u,6454673133771695u }  ;
              uint64_t vb[] = { 9066071611000681104u,1997934710739599u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -3 );
        }
                {
              uint64_t va[] = { 5071896329492196415u,4720635858964020u }  ;
              uint64_t vb[] = { 3758598055640787916u,12552953127405247u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -7 );
        }
                {
              uint64_t va[] = { 18413410920331609388u,6223639009065534u }  ;
              uint64_t vb[] = { 11063026832877662549u,21307634852586051u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -17 );
        }
                {
              uint64_t va[] = { 204221934341094477u,12077039384540444u }  ;
              uint64_t vb[] = { 11985595970574947585u,23646307246946519u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  9 );
        }
                {
              uint64_t va[] = { 14932852042423758232u,18388528454532328u }  ;
              uint64_t vb[] = { 12439556305563360314u,10533371192048036u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  11 );
        }
                {
              uint64_t va[] = { 11051267257580058747u,35352969919313614u }  ;
              uint64_t vb[] = { 4200175737873476313u,2153241148191077u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -1 );
        }
                {
              uint64_t va[] = { 2867404286596367271u,4697926370083079u }  ;
              uint64_t vb[] = { 14662409776399286062u,25833240421652261u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  11 );
        }
                {
              uint64_t va[] = { 8531965848106108535u,19960949431824032u }  ;
              uint64_t vb[] = { 7924043896445582817u,13894976486290320u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -5 );
        }
                {
              uint64_t va[] = { 2578649535727914566u,13567635430947016u }  ;
              uint64_t vb[] = { 15706647602569546849u,3581822765748378u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  7 );
        }
                {
              uint64_t va[] = { 706125068292516971u,9875878095919026u }  ;
              uint64_t vb[] = { 8391395713918834574u,18965775655499049u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -11 );
        }
                {
              uint64_t va[] = { 16151458226253824011u,34077586905697073u }  ;
              uint64_t vb[] = { 4905653246259679526u,4559337051976523u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  3 );
        }
                {
              uint64_t va[] = { 4149953188516585062u,33046252358304192u }  ;
              uint64_t vb[] = { 13448471889591698127u,29628005201470375u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -1 );
        }
                {
              uint64_t va[] = { 5322235528693270402u,31836701016770782u }  ;
              uint64_t vb[] = { 2972153364654132597u,30143389722482950u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  9 );
        }
                {
              uint64_t va[] = { 12464670166398606051u,31843584368617529u }  ;
              uint64_t vb[] = { 2233281557437858271u,30886096531227596u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -3 );
        }
                {
              uint64_t va[] = { 6770679097642168311u,6533658728814607u }  ;
              uint64_t vb[] = { 15464125443975305705u,33317822661547990u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -17 );
        }
                {
              uint64_t va[] = { 5524556343226414762u,35702853161177058u }  ;
              uint64_t vb[] = { 12071850604989412006u,31703159246399142u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  13 );
        }
                {
              uint64_t va[] = { 2077962343660907977u,21264305647956765u }  ;
              uint64_t vb[] = { 10674103797495960131u,23846810612948571u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -19 );
        }
                {
              uint64_t va[] = { 7061669533665264151u,30510691500261278u }  ;
              uint64_t vb[] = { 11608299101791747851u,8313478289717191u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 2902366873640470210u,31738303777797830u }  ;
              uint64_t vb[] = { 2325620595617446776u,14156322812520105u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  15 );
        }
                {
              uint64_t va[] = { 10770006799817164371u,156359724201665u }  ;
              uint64_t vb[] = { 9656182911326751196u,5460471702947992u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -5 );
        }
                {
              uint64_t va[] = { 5342225200474510040u,19573941540932943u }  ;
              uint64_t vb[] = { 15647438873482774438u,14380151169693267u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -11 );
        }
                {
              uint64_t va[] = { 8137763796978276244u,15807762693620438u }  ;
              uint64_t vb[] = { 9647187982270702829u,32099290568840773u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -5 );
        }
                {
              uint64_t va[] = { 3043223836606342269u,28835700604475925u }  ;
              uint64_t vb[] = { 2759742031428302572u,28423631141396478u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  9 );
        }
                {
              uint64_t va[] = { 17791263768803578647u,32376733941759067u }  ;
              uint64_t vb[] = { 11366932212985454262u,29572172651424536u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  17 );
        }
                {
              uint64_t va[] = { 17860165023005833056u,4294559109858584u }  ;
              uint64_t vb[] = { 8626419962047688558u,4848894481072868u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -9 );
        }
                {
              uint64_t va[] = { 4170899966673244042u,35775631654178170u }  ;
              uint64_t vb[] = { 5971152570993645697u,32051893344721220u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -9 );
        }
                {
              uint64_t va[] = { 1526780473220495760u,381417453391803u }  ;
              uint64_t vb[] = { 6265348386743351897u,24393599757473505u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  3 );
        }
                {
              uint64_t va[] = { 64423479179875411u,29373831128901060u }  ;
              uint64_t vb[] = { 10173649649687447541u,34591159731278950u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  3 );
        }
                {
              uint64_t va[] = { 10750086562974298118u,21917451016366233u }  ;
              uint64_t vb[] = { 3244457440482148659u,24996765531817299u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  5 );
        }
                {
              uint64_t va[] = { 377286477170818416u,31739530137756021u }  ;
              uint64_t vb[] = { 269888407066673776u,6099829946426123u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  5 );
        }
                {
              uint64_t va[] = { 9437056142594440098u,13541847104558349u }  ;
              uint64_t vb[] = { 15666664915836720180u,27405965241660970u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  5 );
        }
                {
              uint64_t va[] = { 17907711878800431638u,3080307259343748u }  ;
              uint64_t vb[] = { 10850871668388349649u,3972546368614366u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -1 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint64_t*, const uint64_t*, size_t), dim=24") {
                {
              uint64_t va[] = { 6046300u }  ;
              uint64_t vb[] = { 8754667u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 2164161u }  ;
              uint64_t vb[] = { 6211060u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 1564529u }  ;
              uint64_t vb[] = { 16136117u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 13775062u }  ;
              uint64_t vb[] = { 14032392u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 12458125u }  ;
              uint64_t vb[] = { 13288285u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 8533485u }  ;
              uint64_t vb[] = { 959696u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 12694878u }  ;
              uint64_t vb[] = { 2994905u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 1240133u }  ;
              uint64_t vb[] = { 15920535u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  8 );
        }
                {
              uint64_t va[] = { 10795430u }  ;
              uint64_t vb[] = { 7072498u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 2998769u }  ;
              uint64_t vb[] = { 12623003u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 3523131u }  ;
              uint64_t vb[] = { 14004263u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 15458705u }  ;
              uint64_t vb[] = { 9066326u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 15455637u }  ;
              uint64_t vb[] = { 15863550u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 14271572u }  ;
              uint64_t vb[] = { 611813u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 13148033u }  ;
              uint64_t vb[] = { 11348179u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 12185132u }  ;
              uint64_t vb[] = { 5878699u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 200279u }  ;
              uint64_t vb[] = { 4683763u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 4806355u }  ;
              uint64_t vb[] = { 8933823u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 15804406u }  ;
              uint64_t vb[] = { 1512698u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 961246u }  ;
              uint64_t vb[] = { 11643813u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 3684770u }  ;
              uint64_t vb[] = { 1887057u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 12666118u }  ;
              uint64_t vb[] = { 15345966u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 1140100u }  ;
              uint64_t vb[] = { 15689860u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 8202181u }  ;
              uint64_t vb[] = { 8387936u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 1056836u }  ;
              uint64_t vb[] = { 12045752u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 7953195u }  ;
              uint64_t vb[] = { 16520426u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 8615903u }  ;
              uint64_t vb[] = { 7268494u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 10261289u }  ;
              uint64_t vb[] = { 1651236u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 16470274u }  ;
              uint64_t vb[] = { 150654u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 11259155u }  ;
              uint64_t vb[] = { 11910452u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 5112666u }  ;
              uint64_t vb[] = { 5069298u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 8048803u }  ;
              uint64_t vb[] = { 5487550u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 8105690u }  ;
              uint64_t vb[] = { 12040119u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 6080152u }  ;
              uint64_t vb[] = { 4274781u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 2293324u }  ;
              uint64_t vb[] = { 9052772u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 8343677u }  ;
              uint64_t vb[] = { 8616263u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 5976940u }  ;
              uint64_t vb[] = { 15406104u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 5731105u }  ;
              uint64_t vb[] = { 13780653u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 415756u }  ;
              uint64_t vb[] = { 8130758u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 1834674u }  ;
              uint64_t vb[] = { 12154518u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  10 );
        }
                {
              uint64_t va[] = { 1047636u }  ;
              uint64_t vb[] = { 10083217u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 16626847u }  ;
              uint64_t vb[] = { 8127u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 15525931u }  ;
              uint64_t vb[] = { 13700673u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 11672792u }  ;
              uint64_t vb[] = { 9245175u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 5312334u }  ;
              uint64_t vb[] = { 15318512u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 5447259u }  ;
              uint64_t vb[] = { 2023205u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 10144064u }  ;
              uint64_t vb[] = { 8834651u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 14395300u }  ;
              uint64_t vb[] = { 411074u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 1738538u }  ;
              uint64_t vb[] = { 15332543u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 14715587u }  ;
              uint64_t vb[] = { 5431828u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 5545337u }  ;
              uint64_t vb[] = { 12065005u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 10462971u }  ;
              uint64_t vb[] = { 8012095u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 16605719u }  ;
              uint64_t vb[] = { 13399783u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 5232052u }  ;
              uint64_t vb[] = { 14312120u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 6499395u }  ;
              uint64_t vb[] = { 6716673u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  10 );
        }
                {
              uint64_t va[] = { 7109881u }  ;
              uint64_t vb[] = { 4153086u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 14253590u }  ;
              uint64_t vb[] = { 11752389u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 11978767u }  ;
              uint64_t vb[] = { 13819458u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 9218551u }  ;
              uint64_t vb[] = { 8536024u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 6517213u }  ;
              uint64_t vb[] = { 745982u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  10 );
        }
                {
              uint64_t va[] = { 6760569u }  ;
              uint64_t vb[] = { 7543926u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 5030325u }  ;
              uint64_t vb[] = { 3738677u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 9027446u }  ;
              uint64_t vb[] = { 14185128u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 7340064u }  ;
              uint64_t vb[] = { 6547283u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 13922444u }  ;
              uint64_t vb[] = { 14322492u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 9898362u }  ;
              uint64_t vb[] = { 12899810u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 14997976u }  ;
              uint64_t vb[] = { 2146242u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 15723923u }  ;
              uint64_t vb[] = { 6787788u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 5536190u }  ;
              uint64_t vb[] = { 1608912u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 10418617u }  ;
              uint64_t vb[] = { 10277024u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  10 );
        }
                {
              uint64_t va[] = { 9593830u }  ;
              uint64_t vb[] = { 10201049u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 13085539u }  ;
              uint64_t vb[] = { 13041082u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 1492612u }  ;
              uint64_t vb[] = { 7584866u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 3790071u }  ;
              uint64_t vb[] = { 11667365u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 7479426u }  ;
              uint64_t vb[] = { 14372533u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 923736u }  ;
              uint64_t vb[] = { 12425644u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 6626352u }  ;
              uint64_t vb[] = { 11428307u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 3919488u }  ;
              uint64_t vb[] = { 14111896u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 1527876u }  ;
              uint64_t vb[] = { 10349239u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 10596300u }  ;
              uint64_t vb[] = { 15370577u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 7957355u }  ;
              uint64_t vb[] = { 9119074u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 6845441u }  ;
              uint64_t vb[] = { 9102679u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 14503387u }  ;
              uint64_t vb[] = { 6561514u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 15456552u }  ;
              uint64_t vb[] = { 16298034u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 7432186u }  ;
              uint64_t vb[] = { 3682275u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  8 );
        }
                {
              uint64_t va[] = { 5946709u }  ;
              uint64_t vb[] = { 8596375u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 2626315u }  ;
              uint64_t vb[] = { 1490431u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 7733497u }  ;
              uint64_t vb[] = { 9704347u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 4776570u }  ;
              uint64_t vb[] = { 7890150u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 5865789u }  ;
              uint64_t vb[] = { 10836616u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 6585144u }  ;
              uint64_t vb[] = { 10845316u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 1719030u }  ;
              uint64_t vb[] = { 3228562u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 12415404u }  ;
              uint64_t vb[] = { 8458228u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 13300293u }  ;
              uint64_t vb[] = { 13877604u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 11005818u }  ;
              uint64_t vb[] = { 9635391u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 15063514u }  ;
              uint64_t vb[] = { 2534497u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 5012193u }  ;
              uint64_t vb[] = { 16165465u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 5513018u }  ;
              uint64_t vb[] = { 10000716u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 14917377u }  ;
              uint64_t vb[] = { 6771159u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 743960u }  ;
              uint64_t vb[] = { 1189249u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint64_t*, const uint64_t*, size_t), dim=23") {
                {
              uint64_t va[] = { 7339331u }  ;
              uint64_t vb[] = { 5877368u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -3 );
        }
                {
              uint64_t va[] = { 3441136u }  ;
              uint64_t vb[] = { 3976529u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  9 );
        }
                {
              uint64_t va[] = { 4033056u }  ;
              uint64_t vb[] = { 6317517u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -13 );
        }
                {
              uint64_t va[] = { 2831424u }  ;
              uint64_t vb[] = { 4431213u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 2677265u }  ;
              uint64_t vb[] = { 3553276u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -9 );
        }
                {
              uint64_t va[] = { 4981599u }  ;
              uint64_t vb[] = { 6440381u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 6618072u }  ;
              uint64_t vb[] = { 4116693u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 2488238u }  ;
              uint64_t vb[] = { 534594u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -9 );
        }
                {
              uint64_t va[] = { 7086476u }  ;
              uint64_t vb[] = { 5921651u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -11 );
        }
                {
              uint64_t va[] = { 2768814u }  ;
              uint64_t vb[] = { 3594234u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  3 );
        }
                {
              uint64_t va[] = { 4315677u }  ;
              uint64_t vb[] = { 4246057u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  13 );
        }
                {
              uint64_t va[] = { 2159819u }  ;
              uint64_t vb[] = { 4774976u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  5 );
        }
                {
              uint64_t va[] = { 1076768u }  ;
              uint64_t vb[] = { 4390480u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  7 );
        }
                {
              uint64_t va[] = { 7373690u }  ;
              uint64_t vb[] = { 6777454u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 5440962u }  ;
              uint64_t vb[] = { 1416586u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  7 );
        }
                {
              uint64_t va[] = { 4698399u }  ;
              uint64_t vb[] = { 2345253u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 8143768u }  ;
              uint64_t vb[] = { 1458020u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -7 );
        }
                {
              uint64_t va[] = { 311121u }  ;
              uint64_t vb[] = { 1142358u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  3 );
        }
                {
              uint64_t va[] = { 5233930u }  ;
              uint64_t vb[] = { 6263897u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  7 );
        }
                {
              uint64_t va[] = { 7445127u }  ;
              uint64_t vb[] = { 7017973u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -1 );
        }
                {
              uint64_t va[] = { 2369664u }  ;
              uint64_t vb[] = { 810955u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -1 );
        }
                {
              uint64_t va[] = { 5222839u }  ;
              uint64_t vb[] = { 4765438u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  5 );
        }
                {
              uint64_t va[] = { 1775567u }  ;
              uint64_t vb[] = { 1567246u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -1 );
        }
                {
              uint64_t va[] = { 2300468u }  ;
              uint64_t vb[] = { 1665655u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  3 );
        }
                {
              uint64_t va[] = { 5231545u }  ;
              uint64_t vb[] = { 6268537u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  7 );
        }
                {
              uint64_t va[] = { 4778303u }  ;
              uint64_t vb[] = { 2790232u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -3 );
        }
                {
              uint64_t va[] = { 7663716u }  ;
              uint64_t vb[] = { 5080132u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  5 );
        }
                {
              uint64_t va[] = { 3858032u }  ;
              uint64_t vb[] = { 8034407u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  7 );
        }
                {
              uint64_t va[] = { 1630765u }  ;
              uint64_t vb[] = { 5099345u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -3 );
        }
                {
              uint64_t va[] = { 1368412u }  ;
              uint64_t vb[] = { 4081370u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 800365u }  ;
              uint64_t vb[] = { 4745007u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  9 );
        }
                {
              uint64_t va[] = { 5007782u }  ;
              uint64_t vb[] = { 6143190u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  3 );
        }
                {
              uint64_t va[] = { 7925178u }  ;
              uint64_t vb[] = { 1450244u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -7 );
        }
                {
              uint64_t va[] = { 4771242u }  ;
              uint64_t vb[] = { 7775259u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -3 );
        }
                {
              uint64_t va[] = { 7144024u }  ;
              uint64_t vb[] = { 2099563u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 1912109u }  ;
              uint64_t vb[] = { 1640902u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  5 );
        }
                {
              uint64_t va[] = { 5877074u }  ;
              uint64_t vb[] = { 2424717u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -9 );
        }
                {
              uint64_t va[] = { 4957700u }  ;
              uint64_t vb[] = { 1463189u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -5 );
        }
                {
              uint64_t va[] = { 2127727u }  ;
              uint64_t vb[] = { 221492u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 4440437u }  ;
              uint64_t vb[] = { 6658318u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -1 );
        }
                {
              uint64_t va[] = { 285378u }  ;
              uint64_t vb[] = { 6459927u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -1 );
        }
                {
              uint64_t va[] = { 2472973u }  ;
              uint64_t vb[] = { 1411490u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -1 );
        }
                {
              uint64_t va[] = { 2277306u }  ;
              uint64_t vb[] = { 1604689u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -5 );
        }
                {
              uint64_t va[] = { 2727642u }  ;
              uint64_t vb[] = { 142904u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 2998161u }  ;
              uint64_t vb[] = { 7313571u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  5 );
        }
                {
              uint64_t va[] = { 4790553u }  ;
              uint64_t vb[] = { 1110424u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  3 );
        }
                {
              uint64_t va[] = { 5803756u }  ;
              uint64_t vb[] = { 1705929u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  7 );
        }
                {
              uint64_t va[] = { 2576928u }  ;
              uint64_t vb[] = { 3317953u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -1 );
        }
                {
              uint64_t va[] = { 1291759u }  ;
              uint64_t vb[] = { 5673060u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  3 );
        }
                {
              uint64_t va[] = { 363496u }  ;
              uint64_t vb[] = { 2456279u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -1 );
        }
                {
              uint64_t va[] = { 2255764u }  ;
              uint64_t vb[] = { 3372782u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 3024785u }  ;
              uint64_t vb[] = { 4697671u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -1 );
        }
                {
              uint64_t va[] = { 6296459u }  ;
              uint64_t vb[] = { 6377155u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  7 );
        }
                {
              uint64_t va[] = { 2914612u }  ;
              uint64_t vb[] = { 7157687u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  5 );
        }
                {
              uint64_t va[] = { 3685607u }  ;
              uint64_t vb[] = { 352704u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -5 );
        }
                {
              uint64_t va[] = { 930703u }  ;
              uint64_t vb[] = { 7461264u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -11 );
        }
                {
              uint64_t va[] = { 1031409u }  ;
              uint64_t vb[] = { 7045120u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -1 );
        }
                {
              uint64_t va[] = { 3094495u }  ;
              uint64_t vb[] = { 4424215u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -1 );
        }
                {
              uint64_t va[] = { 6834169u }  ;
              uint64_t vb[] = { 4477514u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 566658u }  ;
              uint64_t vb[] = { 6828103u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  3 );
        }
                {
              uint64_t va[] = { 3207440u }  ;
              uint64_t vb[] = { 517075u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 6559906u }  ;
              uint64_t vb[] = { 866236u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -3 );
        }
                {
              uint64_t va[] = { 5459730u }  ;
              uint64_t vb[] = { 6289696u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  3 );
        }
                {
              uint64_t va[] = { 4292598u }  ;
              uint64_t vb[] = { 1177671u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  3 );
        }
                {
              uint64_t va[] = { 7390043u }  ;
              uint64_t vb[] = { 5074884u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -7 );
        }
                {
              uint64_t va[] = { 2248752u }  ;
              uint64_t vb[] = { 2124397u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  3 );
        }
                {
              uint64_t va[] = { 229123u }  ;
              uint64_t vb[] = { 5347722u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 7301999u }  ;
              uint64_t vb[] = { 2295317u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -3 );
        }
                {
              uint64_t va[] = { 26320u }  ;
              uint64_t vb[] = { 6370258u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  7 );
        }
                {
              uint64_t va[] = { 7225795u }  ;
              uint64_t vb[] = { 14706u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -3 );
        }
                {
              uint64_t va[] = { 2790172u }  ;
              uint64_t vb[] = { 7558856u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -1 );
        }
                {
              uint64_t va[] = { 1304162u }  ;
              uint64_t vb[] = { 393755u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -1 );
        }
                {
              uint64_t va[] = { 857424u }  ;
              uint64_t vb[] = { 6885768u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  7 );
        }
                {
              uint64_t va[] = { 3915328u }  ;
              uint64_t vb[] = { 4884832u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  3 );
        }
                {
              uint64_t va[] = { 4422437u }  ;
              uint64_t vb[] = { 1968631u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -3 );
        }
                {
              uint64_t va[] = { 2881116u }  ;
              uint64_t vb[] = { 4992005u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -5 );
        }
                {
              uint64_t va[] = { 3509384u }  ;
              uint64_t vb[] = { 1432484u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  5 );
        }
                {
              uint64_t va[] = { 7475028u }  ;
              uint64_t vb[] = { 562486u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -1 );
        }
                {
              uint64_t va[] = { 5593571u }  ;
              uint64_t vb[] = { 2156775u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  3 );
        }
                {
              uint64_t va[] = { 3661192u }  ;
              uint64_t vb[] = { 7901372u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -1 );
        }
                {
              uint64_t va[] = { 7734087u }  ;
              uint64_t vb[] = { 3142357u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -1 );
        }
                {
              uint64_t va[] = { 2196372u }  ;
              uint64_t vb[] = { 2395320u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  5 );
        }
                {
              uint64_t va[] = { 8140025u }  ;
              uint64_t vb[] = { 4018468u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -1 );
        }
                {
              uint64_t va[] = { 6018154u }  ;
              uint64_t vb[] = { 3886716u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  5 );
        }
                {
              uint64_t va[] = { 6162949u }  ;
              uint64_t vb[] = { 2513907u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -3 );
        }
                {
              uint64_t va[] = { 1757899u }  ;
              uint64_t vb[] = { 6538438u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 261923u }  ;
              uint64_t vb[] = { 2712285u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -7 );
        }
                {
              uint64_t va[] = { 1588763u }  ;
              uint64_t vb[] = { 5228097u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -5 );
        }
                {
              uint64_t va[] = { 2560407u }  ;
              uint64_t vb[] = { 7403272u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -9 );
        }
                {
              uint64_t va[] = { 2110949u }  ;
              uint64_t vb[] = { 1976783u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 2592537u }  ;
              uint64_t vb[] = { 1054784u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -5 );
        }
                {
              uint64_t va[] = { 7208038u }  ;
              uint64_t vb[] = { 970660u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 558693u }  ;
              uint64_t vb[] = { 2502387u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  1 );
        }
                {
              uint64_t va[] = { 7357435u }  ;
              uint64_t vb[] = { 2398492u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -5 );
        }
                {
              uint64_t va[] = { 3116474u }  ;
              uint64_t vb[] = { 7541679u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -1 );
        }
                {
              uint64_t va[] = { 238367u }  ;
              uint64_t vb[] = { 6567810u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -3 );
        }
                {
              uint64_t va[] = { 2749660u }  ;
              uint64_t vb[] = { 4381250u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -5 );
        }
                {
              uint64_t va[] = { 1607127u }  ;
              uint64_t vb[] = { 2757881u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  3 );
        }
                {
              uint64_t va[] = { 8377421u }  ;
              uint64_t vb[] = { 3029153u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -3 );
        }
                {
              uint64_t va[] = { 269792u }  ;
              uint64_t vb[] = { 3056261u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -5 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint64_t*, const uint64_t*, size_t), dim=150") {
                {
              uint64_t va[] = { 11737518577408387879u,17808897391988780270u,1802541u }  ;
              uint64_t vb[] = { 4583281146538796958u,12841857161008825099u,153622u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 12013951642977079404u,15981306310753874500u,818518u }  ;
              uint64_t vb[] = { 8706978245155546970u,1152799138370689441u,2722357u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -22 );
        }
                {
              uint64_t va[] = { 7359445067576085750u,18292069383902902791u,3742535u }  ;
              uint64_t vb[] = { 12376426915158920094u,12284943596571258367u,3348173u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 18406339589888091607u,2351340513664235495u,4187343u }  ;
              uint64_t vb[] = { 8107774485494314400u,5922704585220261697u,804113u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  10 );
        }
                {
              uint64_t va[] = { 6507248656515947968u,9566027772874991241u,2774424u }  ;
              uint64_t vb[] = { 14528258788627925358u,3884333035394360719u,4126322u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  8 );
        }
                {
              uint64_t va[] = { 11475808995142156819u,14299026767457341755u,3028285u }  ;
              uint64_t vb[] = { 16950549679175690381u,1579077724321526621u,129688u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -18 );
        }
                {
              uint64_t va[] = { 2026898008872295286u,14984835430494289369u,1715241u }  ;
              uint64_t vb[] = { 12554864170006603866u,1839636556719364723u,1775508u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 6057125086163965971u,6879287195672484909u,9573u }  ;
              uint64_t vb[] = { 2286683903503537856u,9619317923354771615u,297599u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 14786939157619564010u,15970408340059388391u,944800u }  ;
              uint64_t vb[] = { 1673932565282949986u,3766686699383283540u,469921u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  12 );
        }
                {
              uint64_t va[] = { 481790409673226104u,8414574898119021814u,2414561u }  ;
              uint64_t vb[] = { 15222562919161284017u,10177997503965652145u,3361406u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -16 );
        }
                {
              uint64_t va[] = { 15466611049710798736u,6255660287231736863u,331073u }  ;
              uint64_t vb[] = { 9210867184465425919u,6847027893089111227u,487263u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 9046432077975101358u,8458203067487705247u,341538u }  ;
              uint64_t vb[] = { 18174059041052660463u,12073051608039530339u,3863757u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 5411600148612969980u,15102864213565841694u,665894u }  ;
              uint64_t vb[] = { 9116761269923752099u,5359214790638722128u,203537u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -32 );
        }
                {
              uint64_t va[] = { 12803102939984387357u,16900072762285244618u,3352494u }  ;
              uint64_t vb[] = { 12263537759047488566u,13512717885766891824u,166431u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -14 );
        }
                {
              uint64_t va[] = { 3764978873622791822u,16336755906563182755u,46453u }  ;
              uint64_t vb[] = { 8109679104335936143u,1420469274112391552u,584420u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 11777960560455413765u,12798713514201993361u,2476317u }  ;
              uint64_t vb[] = { 13585180897612904295u,2752558720952559673u,676753u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -20 );
        }
                {
              uint64_t va[] = { 158384410733374172u,8540881996685161109u,2152779u }  ;
              uint64_t vb[] = { 14666884200631663306u,5079498245454427850u,644460u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  12 );
        }
                {
              uint64_t va[] = { 7464558148151879311u,16188069800230433996u,1601235u }  ;
              uint64_t vb[] = { 5835381606987763254u,14466105995896314823u,2810261u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 15896058581375248366u,17334467196693261037u,549890u }  ;
              uint64_t vb[] = { 9423966191938236277u,13656748453827164189u,856433u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 10436180289909331732u,9239488095827356867u,1019429u }  ;
              uint64_t vb[] = { 17783021355593813511u,12358387670927977203u,1734579u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  14 );
        }
                {
              uint64_t va[] = { 4492691631439389757u,6552878640389743754u,3990841u }  ;
              uint64_t vb[] = { 9128957259637352772u,17780962142602602161u,1364492u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -12 );
        }
                {
              uint64_t va[] = { 6590372948480438183u,1408507284672987020u,1469655u }  ;
              uint64_t vb[] = { 4728320400011792163u,13760945335372238977u,1949254u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 655784575630668298u,14026197019720788852u,1071923u }  ;
              uint64_t vb[] = { 17086799011972814868u,9734768396733558877u,1059993u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 9420701957717063561u,12101217958835929975u,3528994u }  ;
              uint64_t vb[] = { 11534789525540495190u,1118710906700278104u,1969545u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 9568052531174074738u,13958556168600135177u,3166740u }  ;
              uint64_t vb[] = { 1280858409285861239u,318570503366428815u,4013245u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 11704131044069707093u,7902202591343895u,3881484u }  ;
              uint64_t vb[] = { 17749434837471471029u,12951361124112925800u,3840803u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 6561269921829956413u,16721524058100975342u,650854u }  ;
              uint64_t vb[] = { 2968229252020127432u,2855105772678576613u,1827801u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -18 );
        }
                {
              uint64_t va[] = { 6920072956628234312u,4404491368431677677u,1711960u }  ;
              uint64_t vb[] = { 2932159833157747596u,18252406191485211805u,2721497u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  14 );
        }
                {
              uint64_t va[] = { 13234168278599633767u,4375683078605879913u,1951343u }  ;
              uint64_t vb[] = { 9117673386037771290u,15266175320459898214u,1163723u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 1984104122227225912u,15932592270168975320u,2533646u }  ;
              uint64_t vb[] = { 13264682013604969187u,15202349070864784709u,3428315u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -22 );
        }
                {
              uint64_t va[] = { 17638408759122838782u,17857314459778857707u,4015165u }  ;
              uint64_t vb[] = { 4464079882719616865u,216272618162528847u,3809164u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 4331002565690097379u,5838686014817187029u,3975237u }  ;
              uint64_t vb[] = { 6458512766254334922u,1727519668763644480u,3153885u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  8 );
        }
                {
              uint64_t va[] = { 6231528601913114088u,12058736634855110887u,2443268u }  ;
              uint64_t vb[] = { 17597529838610471826u,10227903791489629251u,779934u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  14 );
        }
                {
              uint64_t va[] = { 16428464229769066692u,16919567078906504593u,3183067u }  ;
              uint64_t vb[] = { 7102447156101507156u,16854297323882246617u,1326893u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  28 );
        }
                {
              uint64_t va[] = { 14644413216680444676u,8666292315763108156u,2776122u }  ;
              uint64_t vb[] = { 9305203956290589510u,2460628376126196267u,3315098u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  26 );
        }
                {
              uint64_t va[] = { 9933578184769341094u,4935890499149536494u,411729u }  ;
              uint64_t vb[] = { 8867264224611554232u,13529048280644003081u,2609846u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -42 );
        }
                {
              uint64_t va[] = { 17363431068686192814u,16756084458784770199u,2167756u }  ;
              uint64_t vb[] = { 13769558761854481943u,16399720268513215946u,3855884u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  16 );
        }
                {
              uint64_t va[] = { 10619282749878883449u,14890254054012678649u,1844736u }  ;
              uint64_t vb[] = { 16842716294069452052u,16394007082709483336u,3503357u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -16 );
        }
                {
              uint64_t va[] = { 17838558811851393043u,18015889127697930323u,1214617u }  ;
              uint64_t vb[] = { 15735599640217685514u,14672826744594297505u,3500037u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 6532605557766983624u,1872484372062668673u,316360u }  ;
              uint64_t vb[] = { 4769801904384612804u,11577310929254044000u,1081475u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 14070602132595922825u,9584381379773114492u,1695216u }  ;
              uint64_t vb[] = { 2206240812878434161u,11723498042926081205u,3879487u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -12 );
        }
                {
              uint64_t va[] = { 1210849873286587388u,9740120777550078785u,233402u }  ;
              uint64_t vb[] = { 2444844361603985455u,11645524300197797923u,1242907u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  12 );
        }
                {
              uint64_t va[] = { 8381433831959990916u,6000704963920616263u,3065006u }  ;
              uint64_t vb[] = { 13594566757158865771u,8927249452437672470u,65166u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 9205436372955238307u,18276159892547030122u,3863847u }  ;
              uint64_t vb[] = { 673886193283989740u,11210053812278691063u,488678u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -12 );
        }
                {
              uint64_t va[] = { 18347273247182816470u,6259743577274962545u,1263781u }  ;
              uint64_t vb[] = { 6677931073734429261u,5532206972430846183u,674819u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  16 );
        }
                {
              uint64_t va[] = { 1116806990718472271u,13931789260930748608u,4131217u }  ;
              uint64_t vb[] = { 10197912234419138722u,2273965932836158277u,834451u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 4738625845049085927u,13223557095740922119u,3794403u }  ;
              uint64_t vb[] = { 11001966989884873794u,11706708307424152417u,2120830u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -12 );
        }
                {
              uint64_t va[] = { 3358360846288494285u,17988980168276218620u,3128482u }  ;
              uint64_t vb[] = { 16986986763321642466u,5885357311458465934u,2854633u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  12 );
        }
                {
              uint64_t va[] = { 639506538654078665u,14324544078218192987u,1846893u }  ;
              uint64_t vb[] = { 4814373305936156266u,15859153564062858714u,60433u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  10 );
        }
                {
              uint64_t va[] = { 1728123703380120171u,5666475717543632538u,4059880u }  ;
              uint64_t vb[] = { 4196798939268818107u,4399397084756133686u,96483u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 10764466636811928598u,4752969165263743060u,2788857u }  ;
              uint64_t vb[] = { 10915453606570020616u,16866454557424121514u,2356591u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 15407168008118865436u,3626959717006201256u,3952172u }  ;
              uint64_t vb[] = { 7888349803551892491u,18165930587626885079u,1262280u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 10549325378233357797u,15799809101303177952u,2324891u }  ;
              uint64_t vb[] = { 6647057400501451105u,8175374914633598605u,554009u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 14594585432775545748u,10582462271869959921u,3847529u }  ;
              uint64_t vb[] = { 10685932059579832882u,18168395090020806160u,2116176u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -16 );
        }
                {
              uint64_t va[] = { 9064451196401049555u,6936967351393213202u,788655u }  ;
              uint64_t vb[] = { 17182810335862496336u,17547458627451129261u,192539u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 14297338540171295958u,9554152929622274406u,2330071u }  ;
              uint64_t vb[] = { 12738629645879400455u,9969101239560844730u,1721939u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 5585586607502879031u,7656572992412169224u,2526537u }  ;
              uint64_t vb[] = { 10818841348249024494u,9896194924512477879u,3017589u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -16 );
        }
                {
              uint64_t va[] = { 3060689833890661131u,11004223972855923291u,1506121u }  ;
              uint64_t vb[] = { 8364979838566588989u,11759805268481107964u,3563637u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -20 );
        }
                {
              uint64_t va[] = { 2544649650477071866u,4823008968861911386u,3483307u }  ;
              uint64_t vb[] = { 6744137706402674369u,16688657781202015455u,4140754u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 12341922999785277544u,15898018314714455391u,237448u }  ;
              uint64_t vb[] = { 830072881654078115u,1093869978408713388u,2054785u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 17726437685228666852u,5643210614350318742u,1722230u }  ;
              uint64_t vb[] = { 14357765253459373653u,6929185388395299259u,2523779u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 4933224320595563250u,6415437468284707938u,3943837u }  ;
              uint64_t vb[] = { 13293317582234578356u,7793344503609260537u,1339512u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 11201401264374521073u,11862727883847461943u,4027076u }  ;
              uint64_t vb[] = { 17666836102114713460u,13418937920185311319u,3185177u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 4961362169100179586u,11790992454873285312u,3626825u }  ;
              uint64_t vb[] = { 11859796262141474940u,7803577688645407667u,1483730u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 1161948783817437943u,15052767215295071562u,980173u }  ;
              uint64_t vb[] = { 17389604048845852109u,7840080485830390548u,1213898u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -12 );
        }
                {
              uint64_t va[] = { 6709278989028400941u,8542930291161711108u,3212267u }  ;
              uint64_t vb[] = { 3955460256952323362u,11754668680744557131u,2458127u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 11880707687517430346u,18193984926250276084u,3587041u }  ;
              uint64_t vb[] = { 15328000380649901180u,15780325030228096703u,1878621u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 5538877051112198355u,11220144715302246370u,2575733u }  ;
              uint64_t vb[] = { 16874504384722052606u,7134066373069493052u,604498u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 15558739768204327200u,10755557132185178141u,4149116u }  ;
              uint64_t vb[] = { 8044758957592722876u,11474199049179804260u,1013663u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 2325547344772329482u,11300203658486974921u,3720667u }  ;
              uint64_t vb[] = { 1068880383353692435u,2704690295168027965u,2919824u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 15796445484752487892u,15950799190455106454u,18044u }  ;
              uint64_t vb[] = { 17857877503122623228u,18435527405222071931u,94207u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  10 );
        }
                {
              uint64_t va[] = { 3184628842080974812u,3639058517461618654u,700321u }  ;
              uint64_t vb[] = { 6536648407079851814u,17700864434208211722u,546931u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  8 );
        }
                {
              uint64_t va[] = { 4002278204675106907u,15770127122156297091u,1260594u }  ;
              uint64_t vb[] = { 7300581491755642437u,6317001166612448569u,34956u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 14067938082122032550u,14328907759788938286u,1063444u }  ;
              uint64_t vb[] = { 18220079600296230769u,14939536707178251783u,3238245u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 11728662175763993399u,14533697523934359174u,4152169u }  ;
              uint64_t vb[] = { 1469813365839513046u,4498984085981882962u,1573469u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  10 );
        }
                {
              uint64_t va[] = { 8037908798696215358u,14114812827444270249u,449095u }  ;
              uint64_t vb[] = { 17819003910669486634u,11075326761384134917u,390826u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  8 );
        }
                {
              uint64_t va[] = { 1476591276516378999u,18432372517615485361u,3418187u }  ;
              uint64_t vb[] = { 17910909116977801186u,7557421350427246970u,1275962u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -6 );
        }
                {
              uint64_t va[] = { 2652920500541303185u,7496783816526548581u,3163022u }  ;
              uint64_t vb[] = { 3807178213254697759u,1952096503742145831u,339800u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  20 );
        }
                {
              uint64_t va[] = { 12326253943353309753u,1204619719021013892u,1690683u }  ;
              uint64_t vb[] = { 9373957318335856889u,4257002161807562153u,3681920u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 12591380274290236485u,9345435415540871703u,4162715u }  ;
              uint64_t vb[] = { 17203713103496264022u,12667827627749326965u,1953141u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  8 );
        }
                {
              uint64_t va[] = { 17961905057317230711u,12980842899137591615u,761345u }  ;
              uint64_t vb[] = { 3618071575299599699u,13927971234448736095u,970747u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  4 );
        }
                {
              uint64_t va[] = { 5832639827704192848u,4637854657197179600u,3326823u }  ;
              uint64_t vb[] = { 3824991248799993795u,7769071285701033648u,3443788u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  26 );
        }
                {
              uint64_t va[] = { 15659980325627867019u,10732906586527784763u,759004u }  ;
              uint64_t vb[] = { 13869624340263588630u,7684840468187001774u,1484987u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  0 );
        }
                {
              uint64_t va[] = { 18316079345370010287u,556203031231567795u,1461269u }  ;
              uint64_t vb[] = { 16566983733099563642u,2317737292818185938u,3235261u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 7974362237982941049u,14130091960877008001u,1051147u }  ;
              uint64_t vb[] = { 10926074984192347848u,9363376830580186465u,1674917u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 10504095525383864598u,9102513260943950970u,3834464u }  ;
              uint64_t vb[] = { 6872859082504306841u,13954838193483121766u,2730078u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 12233640677778882967u,2339496797955789048u,816327u }  ;
              uint64_t vb[] = { 9580510203585833375u,10794462956905108068u,1953957u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  28 );
        }
                {
              uint64_t va[] = { 14572419869772112214u,8975809592913575020u,2145308u }  ;
              uint64_t vb[] = { 18432565714898724389u,5550552256170376740u,1166790u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 806667763527979612u,6554587672610447972u,3575492u }  ;
              uint64_t vb[] = { 1422726480871812319u,14791559645757909022u,1167077u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -2 );
        }
                {
              uint64_t va[] = { 329377875798016310u,6650787243084327014u,3326573u }  ;
              uint64_t vb[] = { 314814659591700519u,756687615861331875u,3904295u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  18 );
        }
                {
              uint64_t va[] = { 14886760369897085813u,8662123357950297051u,3776960u }  ;
              uint64_t vb[] = { 8550631944447770059u,8274470851886040303u,1694978u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  6 );
        }
                {
              uint64_t va[] = { 8744250229068274116u,8866539172875218318u,3016631u }  ;
              uint64_t vb[] = { 17975774723668508625u,3048480772923098347u,3086811u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  24 );
        }
                {
              uint64_t va[] = { 3303355958679529655u,3088762798873633442u,3346292u }  ;
              uint64_t vb[] = { 16763002404769466649u,18045816035403342228u,3073198u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -4 );
        }
                {
              uint64_t va[] = { 8889961834202257885u,5605617524493808460u,3997086u }  ;
              uint64_t vb[] = { 2146205764221975950u,18445263805025587168u,2112034u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -10 );
        }
                {
              uint64_t va[] = { 11754488290893166572u,2152109574919381535u,3635554u }  ;
              uint64_t vb[] = { 15309455414422597869u,574411105278070858u,1037263u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  2 );
        }
                {
              uint64_t va[] = { 10440926919681437916u,15992397906304151133u,423439u }  ;
              uint64_t vb[] = { 10542621387863022516u,17739001389696408705u,2107937u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -12 );
        }
                {
              uint64_t va[] = { 11213387014668800764u,5959900167685829483u,1028037u }  ;
              uint64_t vb[] = { 5014777523542158654u,7740176686807496650u,1575455u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 10088203500873788385u,2758498568005622219u,1515752u }  ;
              uint64_t vb[] = { 336942905129248360u,17892279737116403785u,2841385u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  12 );
        }
                {
              uint64_t va[] = { 13814554566974242269u,8298861656897596060u,587489u }  ;
              uint64_t vb[] = { 7834098413850235020u,11594088021456716954u,4107948u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  -8 );
        }
                {
              uint64_t va[] = { 17655314300251067928u,5279704127703902735u,3624546u }  ;
              uint64_t vb[] = { 15835194174835862622u,11484643917068163998u,3079878u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 64 */, uint64_t>(va, vb, l) ==  12 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint32_t*, const uint32_t*, size_t), , dim=64") {
                {
              uint32_t va[] = { 3286118732u,2553310815u }  ;
              uint32_t vb[] = { 973372227u,641271103u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 3460349253u,398749049u }  ;
              uint32_t vb[] = { 2650604403u,18174350u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -12 );
        }
                {
              uint32_t va[] = { 1877770536u,2267011638u }  ;
              uint32_t vb[] = { 914085662u,2814490620u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 666465922u,3766406078u }  ;
              uint32_t vb[] = { 2134066155u,4288230327u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 2545810027u,942027226u }  ;
              uint32_t vb[] = { 484620550u,2448832025u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 2525450172u,3546545326u }  ;
              uint32_t vb[] = { 875204206u,1788145646u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 2302785775u,2707226141u }  ;
              uint32_t vb[] = { 3493373308u,3417209166u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 1926656069u,88088540u }  ;
              uint32_t vb[] = { 713227959u,1530398200u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 2138793483u,1125436276u }  ;
              uint32_t vb[] = { 1295942899u,3284130668u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  16 );
        }
                {
              uint32_t va[] = { 1104076321u,4172612913u }  ;
              uint32_t vb[] = { 490981330u,3048070610u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 2233248447u,3554562179u }  ;
              uint32_t vb[] = { 3397253520u,2743295406u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 2976118474u,1142468047u }  ;
              uint32_t vb[] = { 3203121005u,2277878196u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 3086226541u,2156578810u }  ;
              uint32_t vb[] = { 3700771520u,361335110u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 270075038u,2737871621u }  ;
              uint32_t vb[] = { 2367887222u,1607776292u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 3971897243u,2593673115u }  ;
              uint32_t vb[] = { 2328104429u,4258241235u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 2420204735u,1138431658u }  ;
              uint32_t vb[] = { 1366399131u,3331241041u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 642233799u,1752056624u }  ;
              uint32_t vb[] = { 1032471703u,3518622492u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 3168817223u,3566357110u }  ;
              uint32_t vb[] = { 2486732815u,1330039705u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 3256720306u,1977043792u }  ;
              uint32_t vb[] = { 1761945229u,1721354342u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 2074963875u,1523071002u }  ;
              uint32_t vb[] = { 2787503506u,3304176277u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 1391446574u,4169072634u }  ;
              uint32_t vb[] = { 1292906287u,1898461693u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 4183229127u,34924668u }  ;
              uint32_t vb[] = { 950162144u,1751403336u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 3907657465u,2896493456u }  ;
              uint32_t vb[] = { 3374988502u,201248942u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 77697051u,4245578546u }  ;
              uint32_t vb[] = { 1199254212u,3842374767u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 4016513828u,1184115429u }  ;
              uint32_t vb[] = { 1380462659u,1126849623u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 3525186223u,950025391u }  ;
              uint32_t vb[] = { 3943869842u,671047556u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 3678202534u,1693586068u }  ;
              uint32_t vb[] = { 2008474143u,2449491842u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 4025143670u,3944403377u }  ;
              uint32_t vb[] = { 843692616u,1011278642u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 1096206305u,1841980387u }  ;
              uint32_t vb[] = { 1869508675u,1194362005u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 1096278440u,2870347871u }  ;
              uint32_t vb[] = { 1425736460u,294248436u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 1163577192u,174961503u }  ;
              uint32_t vb[] = { 1963316944u,1518642949u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  10 );
        }
                {
              uint32_t va[] = { 3465665067u,69146103u }  ;
              uint32_t vb[] = { 85757279u,2822880174u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 1217367188u,2816873965u }  ;
              uint32_t vb[] = { 1032828933u,1097995376u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  14 );
        }
                {
              uint32_t va[] = { 2905443409u,3527804194u }  ;
              uint32_t vb[] = { 2381496901u,3723878433u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 4241117654u,69909049u }  ;
              uint32_t vb[] = { 2078538254u,3891593992u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 3994226029u,1725740605u }  ;
              uint32_t vb[] = { 2016766685u,1944885660u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 3273720302u,1911794151u }  ;
              uint32_t vb[] = { 2862719942u,3966227499u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 3632368049u,2589888892u }  ;
              uint32_t vb[] = { 1076822335u,2231171096u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 638641105u,1806743601u }  ;
              uint32_t vb[] = { 3331971481u,2775487460u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 221448771u,1690888155u }  ;
              uint32_t vb[] = { 4052234353u,3093785320u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -12 );
        }
                {
              uint32_t va[] = { 2157720394u,2344842411u }  ;
              uint32_t vb[] = { 2998507956u,4183148808u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 1571830240u,3078797520u }  ;
              uint32_t vb[] = { 3970079796u,243819750u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 1726366806u,3143653643u }  ;
              uint32_t vb[] = { 725405741u,2527492989u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 170228064u,3333610509u }  ;
              uint32_t vb[] = { 2821773169u,837013736u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 2474568317u,229214963u }  ;
              uint32_t vb[] = { 2692150523u,2623483557u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 2021878262u,1101447731u }  ;
              uint32_t vb[] = { 2599775463u,2026715457u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 154412691u,2378164232u }  ;
              uint32_t vb[] = { 1161708514u,171676152u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 938664441u,1945003254u }  ;
              uint32_t vb[] = { 2951934429u,1162484713u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  14 );
        }
                {
              uint32_t va[] = { 3840002691u,2661503281u }  ;
              uint32_t vb[] = { 1194308628u,1061522670u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -18 );
        }
                {
              uint32_t va[] = { 3323218730u,2619914312u }  ;
              uint32_t vb[] = { 2568805924u,2016420218u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  10 );
        }
                {
              uint32_t va[] = { 1951176590u,2327580629u }  ;
              uint32_t vb[] = { 324224700u,1323798509u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 2863037299u,1225721636u }  ;
              uint32_t vb[] = { 3816436666u,2268728549u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 2325194344u,473881776u }  ;
              uint32_t vb[] = { 1778829554u,3102483853u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 676362730u,2242634274u }  ;
              uint32_t vb[] = { 3449544654u,680712135u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 1666989527u,2646673691u }  ;
              uint32_t vb[] = { 2581923313u,559026748u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 1494623483u,4250904118u }  ;
              uint32_t vb[] = { 1880242217u,155636548u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 446631479u,553311443u }  ;
              uint32_t vb[] = { 844529020u,905703803u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  12 );
        }
                {
              uint32_t va[] = { 1798243078u,236568839u }  ;
              uint32_t vb[] = { 2905895623u,1255662098u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 3138035675u,3541606588u }  ;
              uint32_t vb[] = { 2850307769u,4096366715u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 1924987967u,2932264439u }  ;
              uint32_t vb[] = { 3499646222u,506657485u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 120370190u,4278484064u }  ;
              uint32_t vb[] = { 463423771u,713205096u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 2366868187u,1510498737u }  ;
              uint32_t vb[] = { 1642517678u,1359458505u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 3796098950u,2002985313u }  ;
              uint32_t vb[] = { 4127442330u,1724383359u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  20 );
        }
                {
              uint32_t va[] = { 1060917947u,1240714286u }  ;
              uint32_t vb[] = { 3860652514u,2111499351u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  10 );
        }
                {
              uint32_t va[] = { 292489130u,1654997372u }  ;
              uint32_t vb[] = { 1897011040u,3175411515u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 1926511769u,2070731499u }  ;
              uint32_t vb[] = { 2610046929u,1715767875u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  14 );
        }
                {
              uint32_t va[] = { 1861321762u,3152631706u }  ;
              uint32_t vb[] = { 1697925950u,371268970u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 2084597438u,2060158871u }  ;
              uint32_t vb[] = { 3368290455u,3135395909u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 1215264050u,1143671418u }  ;
              uint32_t vb[] = { 2602749277u,2514499746u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 1206663585u,994374312u }  ;
              uint32_t vb[] = { 1286739568u,2564885262u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 2474255496u,3922124641u }  ;
              uint32_t vb[] = { 1044939294u,3899238561u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 41013609u,2969394564u }  ;
              uint32_t vb[] = { 2023755284u,3703771056u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 1021006343u,3344875002u }  ;
              uint32_t vb[] = { 164468945u,2943119482u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 1863757316u,3585743562u }  ;
              uint32_t vb[] = { 553742550u,3752705698u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 1127662762u,2434286147u }  ;
              uint32_t vb[] = { 830256962u,179020625u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 957020550u,2378813715u }  ;
              uint32_t vb[] = { 2205255728u,2578185032u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 1391939993u,2318958572u }  ;
              uint32_t vb[] = { 217657323u,1191888395u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 1677430256u,1558161410u }  ;
              uint32_t vb[] = { 3714005980u,571656603u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 3317665002u,1076427201u }  ;
              uint32_t vb[] = { 3742047376u,744214790u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 3305445807u,310779885u }  ;
              uint32_t vb[] = { 3733624409u,2158525725u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 3832625416u,742653783u }  ;
              uint32_t vb[] = { 2966164688u,2658187808u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 3836887437u,756245796u }  ;
              uint32_t vb[] = { 912809448u,3468786712u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 2714746569u,4089991468u }  ;
              uint32_t vb[] = { 1333697672u,2686151699u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 1929856450u,4099154843u }  ;
              uint32_t vb[] = { 1472902869u,161501842u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 2497572554u,666564644u }  ;
              uint32_t vb[] = { 1606909242u,432962395u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 3349175457u,3987604775u }  ;
              uint32_t vb[] = { 156598686u,4135228688u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -12 );
        }
                {
              uint32_t va[] = { 724033476u,3663152459u }  ;
              uint32_t vb[] = { 2531051508u,2211444056u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 2356622307u,960515616u }  ;
              uint32_t vb[] = { 468279157u,2535270092u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 3940741264u,2621330099u }  ;
              uint32_t vb[] = { 791357541u,62135360u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 2251784066u,1123467723u }  ;
              uint32_t vb[] = { 2707817471u,2742405125u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 56950917u,638109892u }  ;
              uint32_t vb[] = { 3082706514u,2253098716u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 1587139494u,388671185u }  ;
              uint32_t vb[] = { 1344113089u,3275829311u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 3313840625u,2804483538u }  ;
              uint32_t vb[] = { 544225507u,2580910345u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 3179477149u,1572736185u }  ;
              uint32_t vb[] = { 1677565566u,2560390387u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 3247163239u,3035471293u }  ;
              uint32_t vb[] = { 3667058478u,4172212342u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 1249630357u,852731309u }  ;
              uint32_t vb[] = { 3104792485u,1628423641u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 3879527805u,2376927211u }  ;
              uint32_t vb[] = { 3068928684u,3493414998u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 2676224174u,3339992454u }  ;
              uint32_t vb[] = { 3079462509u,2002928451u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 1640414026u,3337289742u }  ;
              uint32_t vb[] = { 936682776u,1215262579u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 391672978u,2327351131u }  ;
              uint32_t vb[] = { 1714121547u,1585105819u }  ;
              size_t l = 64;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint32_t*, const uint32_t*, size_t), , dim=128") {
                {
              uint32_t va[] = { 1404379145u,2382235256u,664647711u,1634002659u }  ;
              uint32_t vb[] = { 2237391232u,3269353579u,117084316u,1743044941u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  20 );
        }
                {
              uint32_t va[] = { 2820583123u,678492060u,2649115847u,3607290640u }  ;
              uint32_t vb[] = { 2010961321u,2939518272u,4140268505u,2633451077u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -12 );
        }
                {
              uint32_t va[] = { 2782953347u,264381737u,3529711816u,3878857488u }  ;
              uint32_t vb[] = { 1519626112u,2695538281u,980020912u,652604551u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 2931847445u,560805250u,3341048403u,818458793u }  ;
              uint32_t vb[] = { 2255423768u,2498661244u,2455599867u,420305502u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 2831873453u,1781509809u,2082549391u,903855358u }  ;
              uint32_t vb[] = { 3854511127u,2094889045u,2120488581u,1668757876u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  12 );
        }
                {
              uint32_t va[] = { 3246344864u,1672215421u,743431495u,1028670102u }  ;
              uint32_t vb[] = { 2059449010u,494345421u,2456121402u,839872701u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -26 );
        }
                {
              uint32_t va[] = { 1211573080u,1664563176u,673290313u,2796610596u }  ;
              uint32_t vb[] = { 4040031103u,718098341u,1207021623u,112855320u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 260714336u,2301358085u,1391048290u,3288878955u }  ;
              uint32_t vb[] = { 3733416203u,367707130u,4127100615u,1151203183u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 3248237165u,3287820897u,2382461917u,3638451395u }  ;
              uint32_t vb[] = { 3914026024u,1971766934u,632402544u,2038191414u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 964159348u,263358558u,3167135921u,3011964132u }  ;
              uint32_t vb[] = { 1977032448u,3033490686u,1250037560u,4051294565u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 2641281725u,386408171u,3594073831u,2523902012u }  ;
              uint32_t vb[] = { 1350797225u,1315709518u,3428858839u,3034218189u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 1090151484u,1729315812u,2263019168u,829149380u }  ;
              uint32_t vb[] = { 3687163885u,111126578u,1735952154u,698681938u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 4212519635u,756789084u,2813669853u,881925828u }  ;
              uint32_t vb[] = { 4147555814u,2187386800u,1415147969u,2984514970u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 4191173106u,2353841313u,121195666u,795312042u }  ;
              uint32_t vb[] = { 2784742022u,3784882023u,4293495739u,1285414137u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -12 );
        }
                {
              uint32_t va[] = { 1898743027u,652150679u,2167250832u,4224671109u }  ;
              uint32_t vb[] = { 2088737605u,3378003978u,3830475406u,594893489u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -16 );
        }
                {
              uint32_t va[] = { 2823710144u,343701981u,3891427031u,2611194982u }  ;
              uint32_t vb[] = { 487603394u,1441692616u,841688426u,262936253u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 679601027u,1434129314u,2929069111u,2711455463u }  ;
              uint32_t vb[] = { 3842390578u,2140658489u,1472495809u,332426091u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 671753189u,2082915048u,2274309729u,1538912368u }  ;
              uint32_t vb[] = { 1347023325u,3925591638u,4182283476u,2291499345u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 2294157797u,3309048381u,1091123519u,3356559668u }  ;
              uint32_t vb[] = { 4207215473u,3295707560u,1491315137u,3586700958u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 4228018863u,1248754778u,3353103159u,3631757365u }  ;
              uint32_t vb[] = { 1801115170u,3156199772u,2348635670u,2075323350u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 1499830849u,3763260986u,2956658844u,3079768771u }  ;
              uint32_t vb[] = { 3635361304u,2574126222u,473884844u,1763857528u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 3507531507u,1673918332u,4115827522u,1545237342u }  ;
              uint32_t vb[] = { 443586037u,2550231816u,3186631144u,884106791u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  14 );
        }
                {
              uint32_t va[] = { 3580906724u,3685547273u,1149934433u,3167739958u }  ;
              uint32_t vb[] = { 1673989515u,2512394355u,2110396695u,4131479763u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 1458449297u,2044641033u,71015923u,2641657049u }  ;
              uint32_t vb[] = { 2867644028u,2515917194u,1757562468u,1045259175u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -22 );
        }
                {
              uint32_t va[] = { 1824823038u,14570687u,2487161375u,3690497580u }  ;
              uint32_t vb[] = { 1267994211u,1068166864u,3719545401u,2372461840u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 2209628551u,4206788040u,221959698u,1778983421u }  ;
              uint32_t vb[] = { 2819258242u,2780497859u,3503804642u,3194201953u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 1775732962u,1963173618u,1027256622u,3623461380u }  ;
              uint32_t vb[] = { 3069252606u,3743105291u,492827531u,2039674220u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 2684442966u,3098462512u,1365409148u,3106214651u }  ;
              uint32_t vb[] = { 1355349275u,774871126u,1907652925u,358895280u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  26 );
        }
                {
              uint32_t va[] = { 3123219064u,4228126117u,2354594498u,2334227607u }  ;
              uint32_t vb[] = { 2587193090u,1843754416u,3106548097u,3015803508u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 4150139117u,979777133u,740481453u,1314421943u }  ;
              uint32_t vb[] = { 1172561502u,637321497u,2095601194u,513615842u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 3665111380u,3556423123u,685810287u,1871729339u }  ;
              uint32_t vb[] = { 3512391555u,689143550u,551251071u,3838842738u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 4121426626u,1248666419u,228549288u,3877038474u }  ;
              uint32_t vb[] = { 2114829207u,3784526475u,2953673719u,1103909700u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -14 );
        }
                {
              uint32_t va[] = { 2466780013u,3824528952u,2324257454u,4008886627u }  ;
              uint32_t vb[] = { 1333794027u,1363618938u,2063576771u,3928565815u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  10 );
        }
                {
              uint32_t va[] = { 1170201830u,303860893u,2267736841u,1202915575u }  ;
              uint32_t vb[] = { 3945002166u,1008115237u,553878580u,1791699329u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  12 );
        }
                {
              uint32_t va[] = { 1349137892u,2031256187u,2541599889u,1012104580u }  ;
              uint32_t vb[] = { 2178843593u,2702173697u,3116626311u,2463403052u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 3582028431u,3091522691u,3077947451u,1352441742u }  ;
              uint32_t vb[] = { 2953329464u,2943170107u,3282557349u,1731505819u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 3154834148u,647119862u,2709311178u,179595413u }  ;
              uint32_t vb[] = { 1383418660u,3973498885u,646972087u,3047110814u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -14 );
        }
                {
              uint32_t va[] = { 4125755941u,3982589690u,744694436u,1024670376u }  ;
              uint32_t vb[] = { 3004821621u,2171317445u,2171365475u,3490035972u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 487964945u,356751066u,4281923154u,2575023836u }  ;
              uint32_t vb[] = { 15826218u,3421088240u,95631031u,3255458994u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -16 );
        }
                {
              uint32_t va[] = { 295950409u,893670768u,2820985651u,967786939u }  ;
              uint32_t vb[] = { 2412419270u,1065601258u,1379001526u,1162411764u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -14 );
        }
                {
              uint32_t va[] = { 4029808551u,775354127u,618849715u,1784042722u }  ;
              uint32_t vb[] = { 2647333328u,2309998178u,2700418989u,1536648855u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 4261565985u,2368741225u,2009500332u,433069309u }  ;
              uint32_t vb[] = { 3099026542u,1718895131u,2517502842u,1616700537u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 2416775693u,1967090631u,1729105076u,4292389287u }  ;
              uint32_t vb[] = { 213430448u,4172681406u,2228203955u,480778954u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -22 );
        }
                {
              uint32_t va[] = { 3876840749u,1702719866u,3179849542u,173283389u }  ;
              uint32_t vb[] = { 3618981952u,3116460174u,1959221087u,3802784268u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -12 );
        }
                {
              uint32_t va[] = { 3146313407u,807322723u,673139588u,2591773832u }  ;
              uint32_t vb[] = { 997667753u,1777554989u,2726638929u,272353869u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 4141566406u,2514967370u,4199257983u,2410879426u }  ;
              uint32_t vb[] = { 2603655485u,233249661u,3856311372u,4121798964u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 1890413708u,3216924982u,3777978984u,2316540325u }  ;
              uint32_t vb[] = { 3661879687u,2349823586u,3331531808u,1459195992u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 2221036386u,454727672u,4121369735u,3561642695u }  ;
              uint32_t vb[] = { 3429394003u,3600407208u,239859036u,358687254u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 1509615418u,1428759728u,1756653944u,579603942u }  ;
              uint32_t vb[] = { 607879637u,2982119075u,622009430u,2194983572u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 2421971511u,948892472u,3716479563u,2228483000u }  ;
              uint32_t vb[] = { 3783190723u,1560452934u,3830601384u,982826124u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 2644255024u,1794543872u,2202505331u,2845332238u }  ;
              uint32_t vb[] = { 3472409723u,608913727u,1243458062u,1426451244u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -14 );
        }
                {
              uint32_t va[] = { 2256755864u,3175524284u,4011758221u,2901338343u }  ;
              uint32_t vb[] = { 1147569679u,3135673770u,1790481391u,3332716249u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  10 );
        }
                {
              uint32_t va[] = { 3480546183u,3528951990u,2294549070u,2519176910u }  ;
              uint32_t vb[] = { 103439896u,71883739u,1685705958u,2921205022u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -20 );
        }
                {
              uint32_t va[] = { 3553571570u,952506821u,2634807636u,3554608979u }  ;
              uint32_t vb[] = { 1015286699u,108153847u,158833973u,4264498919u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 683394216u,3025586281u,649537899u,1992043999u }  ;
              uint32_t vb[] = { 3944954949u,1391038482u,471683655u,693022191u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -12 );
        }
                {
              uint32_t va[] = { 3106154796u,3661086204u,4256026207u,1099712289u }  ;
              uint32_t vb[] = { 3000289857u,139480375u,267314093u,703722002u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -12 );
        }
                {
              uint32_t va[] = { 46850766u,846627385u,1223052048u,1452438593u }  ;
              uint32_t vb[] = { 2278087533u,2370499876u,3302826510u,3297018176u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 1489809265u,3209451015u,1311579413u,1721325512u }  ;
              uint32_t vb[] = { 2674911711u,4102343160u,3692991159u,1536732513u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 593943818u,2285210310u,1751896632u,1756852885u }  ;
              uint32_t vb[] = { 3422776865u,2699701011u,593023778u,2953163028u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 2331658536u,1726553285u,879736527u,1872459190u }  ;
              uint32_t vb[] = { 3850619918u,1460425264u,177004282u,20689821u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 1802381691u,2937560788u,1496801240u,1474437237u }  ;
              uint32_t vb[] = { 3967717470u,72232379u,3725979606u,1355378352u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 2451391412u,2757343721u,4040140689u,3038358486u }  ;
              uint32_t vb[] = { 1808391570u,3490645005u,767522550u,1162883419u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 2961405287u,378157048u,348038730u,566418039u }  ;
              uint32_t vb[] = { 1804391856u,3832181079u,2797471047u,3747576153u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 2143126308u,941693182u,2551752364u,852388075u }  ;
              uint32_t vb[] = { 586484990u,1777230272u,52103190u,2506629174u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -12 );
        }
                {
              uint32_t va[] = { 134250401u,2533691605u,2860526457u,1043152678u }  ;
              uint32_t vb[] = { 2439755822u,3135720922u,720221303u,1177541320u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 780527920u,2197203768u,3922817961u,3105462939u }  ;
              uint32_t vb[] = { 2531760545u,1318633904u,1035560384u,3941014725u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 3133178874u,208530210u,2445999746u,2450347789u }  ;
              uint32_t vb[] = { 1239601286u,3578069045u,630539923u,1229497646u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 1140473667u,3603603892u,2110417091u,1461724803u }  ;
              uint32_t vb[] = { 2092624551u,527086806u,3516639519u,788914935u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  14 );
        }
                {
              uint32_t va[] = { 2647870861u,2193761979u,3077987590u,2401312960u }  ;
              uint32_t vb[] = { 2385581123u,205616782u,213717688u,4081301070u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 1287027619u,2554745116u,283479034u,3957851897u }  ;
              uint32_t vb[] = { 1050218189u,1346678122u,417000532u,227745099u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -12 );
        }
                {
              uint32_t va[] = { 31769001u,2341849314u,623474774u,2541643144u }  ;
              uint32_t vb[] = { 3946989421u,1823658628u,1185401519u,2053449346u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 327940985u,3682676033u,1327681556u,3627443089u }  ;
              uint32_t vb[] = { 2386229349u,964496367u,44058589u,3756408991u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  10 );
        }
                {
              uint32_t va[] = { 2333810622u,234965000u,3486892551u,3607997559u }  ;
              uint32_t vb[] = { 1772077177u,3282139783u,4233176341u,2634614515u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  14 );
        }
                {
              uint32_t va[] = { 2596149382u,2573575764u,729797395u,750880801u }  ;
              uint32_t vb[] = { 3197420661u,3824806788u,3165837761u,341226815u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 383313274u,956869385u,585942138u,346415502u }  ;
              uint32_t vb[] = { 3712167851u,2720498393u,2495988764u,284976781u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  14 );
        }
                {
              uint32_t va[] = { 2925498524u,3202707279u,665714584u,3894889203u }  ;
              uint32_t vb[] = { 3038305382u,3267329333u,1540141553u,3758485241u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  14 );
        }
                {
              uint32_t va[] = { 4153230776u,4146860829u,3477607512u,1607881969u }  ;
              uint32_t vb[] = { 3128944584u,138163638u,4172960034u,2031186969u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 2339820672u,2857828336u,4161244290u,2964235775u }  ;
              uint32_t vb[] = { 3499418350u,4012941363u,2259543298u,833940997u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 757440630u,1308020803u,3220216569u,3679250872u }  ;
              uint32_t vb[] = { 2844677734u,1978321805u,180282885u,4184523571u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 2378246142u,4163117511u,869431325u,2593003450u }  ;
              uint32_t vb[] = { 460217732u,2396840383u,3558023190u,979408755u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 909955144u,3135064072u,2454949108u,527996596u }  ;
              uint32_t vb[] = { 3368162765u,228298524u,2782807588u,3852998548u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 2358802196u,69762331u,1954998196u,3464075216u }  ;
              uint32_t vb[] = { 734592371u,3122733594u,3221336591u,1957322428u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 2699369287u,631467017u,1467792304u,1193106883u }  ;
              uint32_t vb[] = { 1680029832u,3806345050u,3369412457u,3735649871u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 3321860407u,2512686114u,165956358u,1927214229u }  ;
              uint32_t vb[] = { 2868294524u,4115980941u,1376642822u,1219127197u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  16 );
        }
                {
              uint32_t va[] = { 2879847096u,3726799127u,993376405u,936486702u }  ;
              uint32_t vb[] = { 1436490549u,458702468u,3429759685u,1741462031u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 2934619818u,75263160u,543808898u,2519399600u }  ;
              uint32_t vb[] = { 1233332740u,4063135918u,1806171259u,3651011697u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 1732296436u,1720217287u,1260189503u,2034903838u }  ;
              uint32_t vb[] = { 2933298097u,1837324947u,298055817u,1287598497u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 3693366084u,3364020703u,3470880384u,3141122407u }  ;
              uint32_t vb[] = { 3058869655u,3099001222u,1637905217u,1486870148u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 82609984u,971775949u,938915004u,3961696617u }  ;
              uint32_t vb[] = { 2281533969u,733111386u,2257607077u,937464361u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 1014336010u,3784379477u,2844070392u,2334596462u }  ;
              uint32_t vb[] = { 3284637912u,3697816431u,3287710066u,3306952808u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 936607512u,1559907571u,2072842634u,892941271u }  ;
              uint32_t vb[] = { 2581101337u,1206418429u,3726014124u,3008865039u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  18 );
        }
                {
              uint32_t va[] = { 2363379422u,3812641882u,4148397131u,3500916164u }  ;
              uint32_t vb[] = { 1366679369u,1280109706u,3205835046u,867963137u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 2667228298u,1786860246u,734134086u,4004561859u }  ;
              uint32_t vb[] = { 4273622806u,2059894986u,1950488687u,2293341636u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  20 );
        }
                {
              uint32_t va[] = { 1316654164u,1717780519u,1738394455u,516376125u }  ;
              uint32_t vb[] = { 1346269431u,1398755902u,3612168343u,664261113u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  16 );
        }
                {
              uint32_t va[] = { 1708150123u,914383131u,2864946854u,410410559u }  ;
              uint32_t vb[] = { 2820732667u,1487770441u,4031297557u,3276686340u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 574556350u,2557549113u,1026711829u,2757856267u }  ;
              uint32_t vb[] = { 2002696837u,1857215571u,3638649868u,2149843876u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 2774698735u,3461150144u,3328611383u,1479310518u }  ;
              uint32_t vb[] = { 70428722u,2919357974u,194983082u,1669194422u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  10 );
        }
                {
              uint32_t va[] = { 3891707930u,861893006u,2006592106u,875590470u }  ;
              uint32_t vb[] = { 1723585u,1054456543u,2032101524u,3367182217u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 3434074081u,3917556104u,1462334925u,3032576813u }  ;
              uint32_t vb[] = { 1123785054u,3418794267u,4248815024u,1031726153u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 150584963u,2712030853u,2348595326u,1488406163u }  ;
              uint32_t vb[] = { 1705072917u,2598308370u,910258788u,2554821040u }  ;
              size_t l = 128;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint32_t*, const uint32_t*, size_t), , dim=256") {
                {
              uint32_t va[] = { 4228804420u,2487532161u,2940595688u,4114759396u,4175236286u,118094074u,1944352068u,1983418688u }  ;
              uint32_t vb[] = { 840046591u,1276725477u,3836317180u,4039988703u,3277307403u,2116313270u,3465729785u,2188261451u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -12 );
        }
                {
              uint32_t va[] = { 143756095u,2267344348u,1351768600u,660780005u,2620871771u,3260671797u,638737735u,147263641u }  ;
              uint32_t vb[] = { 265413012u,503717635u,1669507481u,1342836625u,3763930890u,2123742266u,1610698502u,1056047633u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  18 );
        }
                {
              uint32_t va[] = { 2353461693u,4235122710u,3364905046u,685586352u,529175976u,2827863153u,1912640890u,908354820u }  ;
              uint32_t vb[] = { 414948225u,3291401012u,3276002076u,667325616u,2567926017u,2035672251u,54709961u,3764736961u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  32 );
        }
                {
              uint32_t va[] = { 3039883045u,510804148u,101846265u,3656939353u,808345320u,2426549401u,852470534u,3425582727u }  ;
              uint32_t vb[] = { 4214328969u,2241133406u,1408140306u,2659692848u,3794638078u,629107151u,3372436811u,1712702289u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -32 );
        }
                {
              uint32_t va[] = { 444161107u,365118712u,902590856u,154741669u,318860726u,4168817745u,1338787824u,2799949922u }  ;
              uint32_t vb[] = { 2542031708u,786994674u,3654820990u,1765065233u,3344355092u,3704701096u,3958697951u,3824909022u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  10 );
        }
                {
              uint32_t va[] = { 2821041851u,1105354453u,1531293724u,1358313255u,2141062399u,314805663u,475482712u,1250017272u }  ;
              uint32_t vb[] = { 3647657225u,4056296084u,3953507625u,665910778u,3861193753u,1238169905u,3093466322u,4236811455u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 3399887730u,883916280u,3314036191u,2682222514u,1350384891u,2848750654u,4239641935u,2143985916u }  ;
              uint32_t vb[] = { 4220427662u,2361119805u,4149614431u,891264768u,2106111946u,1423211619u,1685488581u,1816851998u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  26 );
        }
                {
              uint32_t va[] = { 3936286572u,1382939658u,2312110389u,2440993386u,3453436164u,3931689816u,631220611u,1201911494u }  ;
              uint32_t vb[] = { 4004682170u,1890712628u,4178612992u,1544235187u,1549597965u,1820721358u,2626769529u,2421743262u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 3631749866u,1449041836u,27172552u,397829358u,1041360086u,1310465551u,762390937u,1405571356u }  ;
              uint32_t vb[] = { 198437600u,1542673986u,2764307748u,513408272u,174111003u,4132361671u,1322458177u,2753394087u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -18 );
        }
                {
              uint32_t va[] = { 1481375804u,2165863737u,3696983152u,456222575u,1680825447u,3211948773u,3098960057u,154329505u }  ;
              uint32_t vb[] = { 868677608u,1781301356u,3242010487u,622983844u,3153684632u,1362519104u,2514372584u,378809166u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -28 );
        }
                {
              uint32_t va[] = { 2158593957u,4090302152u,3410668291u,59611903u,3675750916u,639652112u,2788123230u,748173094u }  ;
              uint32_t vb[] = { 138141956u,3196437971u,567630561u,452356368u,674234323u,4134245085u,2844159219u,2010822314u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -12 );
        }
                {
              uint32_t va[] = { 2152404196u,3021143835u,1654375120u,947521257u,1611486680u,770967736u,243475758u,356885625u }  ;
              uint32_t vb[] = { 3809699777u,146494990u,1822051301u,3000244333u,806571645u,1331795172u,2924330823u,2730807179u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  20 );
        }
                {
              uint32_t va[] = { 2682948840u,1255019224u,3643847047u,4231932939u,2393726782u,3530915079u,438200676u,3907893572u }  ;
              uint32_t vb[] = { 131798821u,288479104u,541772683u,4211417406u,2885979646u,3475150257u,2862403046u,2892283401u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  14 );
        }
                {
              uint32_t va[] = { 1546703782u,1128680718u,122090762u,308475763u,1250183100u,626433877u,4243905345u,1648312833u }  ;
              uint32_t vb[] = { 2973145376u,226214430u,3867646515u,4108572372u,1046168988u,801869300u,959815762u,3363875385u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  18 );
        }
                {
              uint32_t va[] = { 2304719087u,2240109070u,1238465015u,2861965535u,511613946u,508263779u,1033578851u,517075979u }  ;
              uint32_t vb[] = { 4054684193u,1058053386u,483719760u,2871237807u,2680537558u,1148077890u,266476553u,3407028946u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  10 );
        }
                {
              uint32_t va[] = { 3160249716u,4054178582u,2594065637u,3351612340u,624855012u,976281980u,4202346107u,3150737296u }  ;
              uint32_t vb[] = { 1113785439u,844687741u,3487007668u,2007509022u,3248075493u,1433858180u,2494204944u,4073377836u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -12 );
        }
                {
              uint32_t va[] = { 774734586u,4157730671u,3331894569u,1143402678u,2936034739u,2585195067u,397261068u,3915173960u }  ;
              uint32_t vb[] = { 527558639u,3691830402u,1517954179u,4278703413u,2853114952u,2691312709u,2179578957u,2518870373u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 451715069u,1648345865u,3939272000u,638782258u,3309622915u,1466590407u,2814103555u,4218455299u }  ;
              uint32_t vb[] = { 3424185661u,481169488u,2136359485u,2498507922u,1834757648u,310253392u,3545925419u,184984095u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 986427836u,1090384785u,2376502776u,3735324254u,1581333970u,406159994u,2844055988u,1483375140u }  ;
              uint32_t vb[] = { 855996255u,4292495125u,1177723175u,2433240013u,3423788306u,2282800234u,2081772385u,1806469083u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 3874211628u,1409954475u,3523902193u,365586258u,3293459958u,4122960734u,199597240u,2716369526u }  ;
              uint32_t vb[] = { 2535550373u,2604813482u,1971554398u,2930304977u,3610160522u,2796438623u,328175319u,365317460u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  24 );
        }
                {
              uint32_t va[] = { 2678907305u,2075138827u,2228639042u,1644669181u,3478839022u,2202176026u,319493329u,605153743u }  ;
              uint32_t vb[] = { 806893418u,370816868u,564627259u,3145281543u,3550998358u,2279430219u,333166310u,1882952445u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 3555842202u,484786923u,1342942654u,1668605744u,3852507561u,707615607u,2352117744u,3477547895u }  ;
              uint32_t vb[] = { 1077671298u,2240718574u,1935762857u,3901623868u,1474244081u,4136572627u,2340291143u,3505775118u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 1479008308u,3325908165u,1913875913u,3514043225u,1220426499u,1383842468u,150504111u,1836452342u }  ;
              uint32_t vb[] = { 559172209u,753995534u,3960799677u,3493515203u,2262948128u,3972778720u,101686259u,3196289744u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 2149054778u,1438763364u,3834677747u,1646737792u,1742863355u,2521752465u,951350156u,3866334787u }  ;
              uint32_t vb[] = { 2545552199u,1671717826u,1620138362u,554086434u,1148606488u,2209921713u,1392195322u,3081575643u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  18 );
        }
                {
              uint32_t va[] = { 1294575468u,946832309u,228673240u,1640818082u,2916729453u,2149925242u,1713625960u,867781645u }  ;
              uint32_t vb[] = { 1849374756u,4147807103u,2240859325u,828972971u,2434524226u,3342311710u,3938134360u,10731696u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  38 );
        }
                {
              uint32_t va[] = { 3860002672u,1011818956u,1206171107u,2373922477u,523045477u,2448268176u,10030440u,1955912331u }  ;
              uint32_t vb[] = { 1008446337u,1351935697u,3842364543u,3950396743u,940641351u,428679354u,2141223410u,2567310584u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 2497424022u,862169242u,4054546757u,3430679280u,2910319615u,2804481757u,1248696429u,868066991u }  ;
              uint32_t vb[] = { 1667815504u,3875556958u,2791674565u,2290403100u,1518200423u,80308194u,4148727777u,353844059u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -14 );
        }
                {
              uint32_t va[] = { 601784794u,3962439233u,3909646683u,1858990933u,3679688309u,2055136535u,3159732384u,357806u }  ;
              uint32_t vb[] = { 1343802148u,3450869729u,3424673138u,3647772037u,882146318u,701433522u,409925353u,785133854u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  10 );
        }
                {
              uint32_t va[] = { 2828135747u,1738640427u,2740030170u,493772u,2523132291u,2126546337u,1482303495u,287439660u }  ;
              uint32_t vb[] = { 1860620741u,3764037008u,7608810u,2800814049u,372824195u,1881614414u,3157435305u,299318033u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 2761274738u,27798223u,1934818579u,2451585146u,2263522198u,4267312915u,1804107111u,1232624040u }  ;
              uint32_t vb[] = { 3993330711u,248189756u,2127483904u,135669724u,4172577615u,2042569819u,3888784987u,2764823148u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 4136715432u,3755579298u,1157425490u,3536106695u,3311118725u,2881367463u,662606506u,3809439424u }  ;
              uint32_t vb[] = { 1377435229u,2375860331u,2041461360u,3544685940u,3813370670u,3137348193u,4114332902u,2323423499u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  18 );
        }
                {
              uint32_t va[] = { 136198512u,4208334639u,1053432675u,2250471787u,659316438u,1878733921u,2367846407u,2080014950u }  ;
              uint32_t vb[] = { 2377948139u,2965846869u,3625102774u,3347946926u,2112626859u,532665441u,2387486068u,986773928u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  18 );
        }
                {
              uint32_t va[] = { 1440783184u,1772240295u,2212354107u,4177615284u,1992166303u,574269373u,652825220u,1817082487u }  ;
              uint32_t vb[] = { 1817647794u,4145947745u,3632302627u,701777729u,1631903206u,746968892u,1381612918u,2615152200u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -28 );
        }
                {
              uint32_t va[] = { 1607414703u,1303960718u,3300600864u,1632779570u,3803075232u,3688270255u,4150566097u,855269344u }  ;
              uint32_t vb[] = { 2768781981u,1333925941u,2184184047u,3027310434u,2000867795u,996129757u,4032992262u,458613650u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  14 );
        }
                {
              uint32_t va[] = { 768255506u,3339694290u,4150566478u,3911095972u,1820499931u,1697354016u,3125163073u,3093546402u }  ;
              uint32_t vb[] = { 1221541316u,148976141u,3047942674u,1310321964u,4279005410u,2170817797u,4222689330u,885863908u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -14 );
        }
                {
              uint32_t va[] = { 997558982u,3190869248u,2014636773u,3097647397u,1275501675u,4098354189u,2642884529u,3996831649u }  ;
              uint32_t vb[] = { 1495166525u,2135571971u,4270840240u,3066520847u,214471171u,3175321549u,2051697374u,3973807671u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  12 );
        }
                {
              uint32_t va[] = { 486227109u,2130731067u,80857022u,1621795219u,485392936u,2763194851u,3044469417u,735934263u }  ;
              uint32_t vb[] = { 3490728215u,2083376474u,1969905868u,573419878u,108618683u,3494029876u,1651394061u,2343995578u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -14 );
        }
                {
              uint32_t va[] = { 4256938026u,276345243u,995788734u,3842421388u,1925558050u,4101616829u,2236024995u,1304095794u }  ;
              uint32_t vb[] = { 2835520221u,1083641950u,2636573860u,3020879789u,3811415842u,2090231524u,1339037930u,1212114189u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  30 );
        }
                {
              uint32_t va[] = { 743293614u,3348917381u,3506529490u,1819542592u,2564601909u,1512789676u,872339322u,1973227468u }  ;
              uint32_t vb[] = { 438163281u,1548116070u,265212657u,1759732528u,32290984u,741593178u,2493372716u,3417432493u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 817624324u,3822486741u,2293793878u,3762070483u,4019791789u,3024163917u,600964871u,972136414u }  ;
              uint32_t vb[] = { 3870123025u,2515815919u,3810914611u,640660740u,1118023638u,136546449u,133149304u,645479796u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 2066862122u,3721887922u,3068760510u,2091088751u,820159699u,1517150255u,1622452073u,2325293321u }  ;
              uint32_t vb[] = { 2581530308u,2678748164u,96951501u,1584170825u,1404899103u,2506219189u,3529822078u,3028038725u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -28 );
        }
                {
              uint32_t va[] = { 1991732167u,3085546198u,602050852u,336561950u,1078346726u,486079556u,879626115u,3767651866u }  ;
              uint32_t vb[] = { 2643303164u,3623471807u,3009656402u,2374174364u,2818270376u,1865435192u,2376159973u,1448269796u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 435856170u,3057552050u,4218832831u,3141432677u,272334137u,887883735u,3563039621u,3934050651u }  ;
              uint32_t vb[] = { 3907547279u,1620978168u,2580394175u,3763892195u,520610022u,2023094480u,3093232722u,3007595958u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 3233470227u,2133929447u,1785469986u,775829886u,1031486555u,2057420790u,515626801u,4034380453u }  ;
              uint32_t vb[] = { 92651706u,1480534626u,2093574860u,1426253659u,2849021449u,2803009282u,1059005981u,2778705788u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 4202052533u,388201003u,844702317u,1279787737u,3734820717u,1344654230u,1641298908u,1326697849u }  ;
              uint32_t vb[] = { 2372570555u,1327143520u,2635453367u,908615859u,3929885010u,4059121837u,3392699293u,3922129541u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 727002519u,2971916834u,251709719u,847307262u,2861796918u,3246394703u,2159340949u,3158528953u }  ;
              uint32_t vb[] = { 744879063u,2443900476u,82971474u,875071533u,109594023u,1378188160u,2680348127u,210085703u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  16 );
        }
                {
              uint32_t va[] = { 2436981492u,3276923870u,3502829741u,4104166826u,1272335192u,3950606248u,2891586850u,4226960128u }  ;
              uint32_t vb[] = { 1629404565u,3875162624u,3550942104u,777917540u,4261789477u,3691919606u,3628324434u,2624054428u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -46 );
        }
                {
              uint32_t va[] = { 3014936131u,4225051424u,4149882416u,2362548166u,2113957916u,74295839u,590519642u,2684413489u }  ;
              uint32_t vb[] = { 2365065464u,621586783u,1322647631u,1682322486u,4007643223u,3541094610u,1667185265u,955857347u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -46 );
        }
                {
              uint32_t va[] = { 3326095430u,4279814287u,278974921u,3288558931u,1655737809u,232943290u,2031835239u,3591800937u }  ;
              uint32_t vb[] = { 2290314300u,632396043u,1740953122u,3235653282u,2707038140u,3248508908u,767481432u,974520262u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 3206904569u,3926243841u,3019397532u,1604399332u,3151283603u,3748615803u,58674055u,102657448u }  ;
              uint32_t vb[] = { 2721276051u,2928847534u,1727448775u,2872329927u,852381039u,418253261u,2444339074u,1911389232u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 1099315402u,1620744698u,2209016883u,1000939162u,2953096766u,2665703694u,4093618653u,329374677u }  ;
              uint32_t vb[] = { 4240323965u,2037616240u,2262179151u,1082052021u,1861937362u,2765976596u,3638974722u,2420773408u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -36 );
        }
                {
              uint32_t va[] = { 2438967159u,1472372416u,381629851u,670706278u,1906399139u,173457279u,3763890381u,199859533u }  ;
              uint32_t vb[] = { 4144920671u,2433265619u,2965818346u,231986063u,127306388u,439703501u,3742763336u,256775166u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 2923450635u,3174009829u,14446405u,1499784350u,2010392144u,1091179269u,4115893841u,2528643075u }  ;
              uint32_t vb[] = { 2174281847u,1588425717u,2490165720u,2835282413u,3795237971u,7840888u,4186347643u,153559487u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 4142102266u,57796623u,2416122756u,1428364460u,3488066114u,968015754u,1697711722u,2110960337u }  ;
              uint32_t vb[] = { 1971100532u,2138076819u,565679946u,112253268u,3244006194u,3178021170u,2827544314u,882910312u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 745906057u,1925862599u,4243052643u,2399909364u,511991456u,3058284491u,2966470598u,605109063u }  ;
              uint32_t vb[] = { 432042121u,1175200331u,4093507845u,3192647962u,1327646691u,3937628406u,635534191u,3462028998u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  20 );
        }
                {
              uint32_t va[] = { 1740609496u,662503703u,1606107429u,3387968257u,2760491786u,1540654556u,3980996626u,3430415734u }  ;
              uint32_t vb[] = { 2899608610u,3879876105u,186078889u,428328110u,3063395380u,1329356245u,827217278u,2126561043u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 2579600870u,2432803888u,317903262u,2978847857u,2495871894u,85083735u,1675842755u,72597u }  ;
              uint32_t vb[] = { 407570956u,3638659699u,1509177012u,952384503u,647499749u,1317126568u,3962708996u,1929964686u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 2977310123u,4024079275u,1010830614u,4053134436u,2693093126u,1453705114u,1083837882u,337662137u }  ;
              uint32_t vb[] = { 735625521u,3070849521u,2162983316u,2464261438u,2152539106u,1702354133u,2465031934u,60080302u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  12 );
        }
                {
              uint32_t va[] = { 561722643u,2320933117u,787756072u,74367841u,2561649820u,980880795u,3388670958u,555140297u }  ;
              uint32_t vb[] = { 3168510518u,2364736401u,4045083812u,1321166831u,2847887686u,3413725009u,2623533490u,1443713553u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 2923583747u,2973021610u,2660833229u,2676617847u,4112304843u,3348821751u,3662087885u,4000968998u }  ;
              uint32_t vb[] = { 727856643u,3727818014u,94656998u,3768818180u,2360050744u,863248900u,2591610179u,868372039u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -24 );
        }
                {
              uint32_t va[] = { 3189440371u,1683030239u,3156963776u,386573046u,580604393u,4257622979u,1735734551u,3020293599u }  ;
              uint32_t vb[] = { 3043293409u,1585025237u,885517688u,767854633u,963587180u,3008627630u,3198085065u,763553842u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 3165164778u,3389651630u,745804407u,3263513883u,3668493755u,1614575235u,2154008417u,2037997188u }  ;
              uint32_t vb[] = { 1182746579u,1893493585u,337360678u,2042483395u,3696966383u,3038225158u,2291096728u,1347551278u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -34 );
        }
                {
              uint32_t va[] = { 1567998066u,2908741381u,539456974u,65527907u,3157621623u,3632313129u,2289913403u,1189720168u }  ;
              uint32_t vb[] = { 3845190828u,1960041877u,4156695216u,3331600032u,2216647963u,1998629056u,3219424972u,1574453159u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -12 );
        }
                {
              uint32_t va[] = { 758584955u,1801673721u,3438514219u,262562255u,3163687168u,3453705600u,1388983645u,2649355863u }  ;
              uint32_t vb[] = { 850886237u,148296337u,2398733990u,3343898119u,83743745u,3839280508u,3376132774u,2184993111u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 219486757u,562189165u,239408240u,10866169u,3239283051u,134920651u,3988010653u,1236808713u }  ;
              uint32_t vb[] = { 3981023993u,346123987u,3745985061u,790260985u,4250871988u,2827756101u,990348766u,1152767984u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 3064531076u,1335699339u,986979317u,734489365u,2714131290u,1606436188u,298037571u,1686366711u }  ;
              uint32_t vb[] = { 1952688383u,3274488722u,2102858032u,861459429u,1785321394u,2311186951u,2666267344u,2741498026u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 1462341006u,3055072428u,2631287313u,1632432072u,3186552425u,3853708059u,221149845u,3676811821u }  ;
              uint32_t vb[] = { 941098119u,3321968597u,2648754813u,3143683058u,39994543u,1753587714u,480382284u,1180755482u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  16 );
        }
                {
              uint32_t va[] = { 3399811100u,594835707u,1571221129u,887088916u,3058230995u,3123441307u,2311572707u,349166787u }  ;
              uint32_t vb[] = { 2261783750u,618197465u,2579505694u,3107705794u,3851117404u,3285891808u,719840516u,2795642681u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 3578783159u,4289826306u,1322250827u,1823602684u,2258806092u,2458347622u,342760119u,1566048114u }  ;
              uint32_t vb[] = { 920738125u,963778723u,4172109226u,4060951354u,4287795646u,3586472779u,1087268694u,2280497521u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 3193894914u,3750124443u,3994973948u,1080237397u,3772894708u,1797361327u,895413002u,3724743160u }  ;
              uint32_t vb[] = { 1639253994u,2531425412u,2659568864u,3638085638u,2451954453u,1900445836u,3760472530u,3770776978u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -16 );
        }
                {
              uint32_t va[] = { 2120742862u,3247699076u,1803179462u,896412730u,52819478u,4082133932u,32901012u,4151562377u }  ;
              uint32_t vb[] = { 906885424u,513231537u,2634351083u,417913794u,3740574151u,350003906u,1869619014u,1556315369u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -20 );
        }
                {
              uint32_t va[] = { 659095019u,973678121u,2714772126u,3361596144u,2399422096u,1018773122u,3655352977u,3780985434u }  ;
              uint32_t vb[] = { 1019457827u,1156631528u,1495816136u,3726935671u,1431221648u,2211705772u,951973274u,715067538u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -14 );
        }
                {
              uint32_t va[] = { 315070717u,3980616562u,3417079411u,4049721646u,1030454161u,3637291848u,2981299159u,1324225022u }  ;
              uint32_t vb[] = { 3936932158u,4210752886u,1315648983u,2620134799u,2657738190u,1134169220u,2107657489u,581848510u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  12 );
        }
                {
              uint32_t va[] = { 2624699818u,2113408931u,2307820714u,1341957766u,3851500887u,3910252313u,3313394172u,796403451u }  ;
              uint32_t vb[] = { 3269173237u,2645111907u,2613942927u,1562582270u,2127064875u,3985145306u,175768999u,286577729u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  10 );
        }
                {
              uint32_t va[] = { 4216369934u,3053874649u,369214235u,2741007939u,3668436331u,509773375u,2066897458u,613041472u }  ;
              uint32_t vb[] = { 2965905608u,1900688100u,2930558800u,3910237916u,2644127237u,3136301329u,1664770350u,4247280122u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 3876738286u,136844092u,14159013u,2420369740u,2456250439u,853450472u,1472739579u,3223730779u }  ;
              uint32_t vb[] = { 4247490900u,745316900u,990948812u,3453229066u,1230668544u,498795173u,952687272u,1670461557u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 3359272000u,3688989235u,909824575u,604207200u,2260321516u,1175880460u,631890173u,516504681u }  ;
              uint32_t vb[] = { 3767931291u,2157443531u,770539541u,498857908u,2483088123u,395658776u,3078517009u,2260605791u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 2645079384u,3113117233u,694209324u,2830693058u,2969569259u,801029264u,1415904519u,2113911374u }  ;
              uint32_t vb[] = { 2880572127u,2115090385u,26650358u,4086244938u,2837752621u,360047762u,1755189659u,3141305060u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 2633633641u,2449682091u,3465425650u,1167353734u,22441485u,2188403170u,4264444228u,1062617953u }  ;
              uint32_t vb[] = { 1938950418u,4173050362u,2215410145u,1328197266u,3219471193u,941344402u,3054111038u,2812996240u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 130223856u,1131077589u,1675846781u,2954676327u,3294744405u,3313308897u,615316832u,1507288138u }  ;
              uint32_t vb[] = { 1518773151u,2268043102u,2269440102u,1689497661u,182180418u,1456805317u,4014507129u,2046168136u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  24 );
        }
                {
              uint32_t va[] = { 1640794723u,293189260u,124432173u,2100063497u,2478993856u,2188713274u,1051117735u,3834466659u }  ;
              uint32_t vb[] = { 3518619730u,2647875313u,2556871686u,544560909u,4088859128u,1024587075u,348474609u,2765281649u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  16 );
        }
                {
              uint32_t va[] = { 1394073399u,3322725305u,972598106u,628958619u,421318069u,2051118977u,1018286588u,4208312878u }  ;
              uint32_t vb[] = { 1230088432u,892309841u,3132570271u,2591196720u,91627848u,1273914097u,1526687135u,3722634329u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -22 );
        }
                {
              uint32_t va[] = { 2049126874u,2083340729u,1787720948u,3786641014u,1626734711u,3807792318u,1409366380u,3544707535u }  ;
              uint32_t vb[] = { 1117097496u,3180424u,4020760116u,4275969955u,3753330517u,3315464029u,2074800302u,1346930228u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -16 );
        }
                {
              uint32_t va[] = { 662348223u,3303436046u,4100789244u,1852846284u,1268468218u,3386688215u,130553470u,1514956481u }  ;
              uint32_t vb[] = { 2443920643u,2517007820u,3340919246u,2119530922u,3266854582u,124063578u,351647808u,1893052223u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 3603423343u,1398063777u,1677510246u,680378193u,1789410315u,3774993086u,908213516u,4253735529u }  ;
              uint32_t vb[] = { 3500384254u,1320380916u,4095792235u,1982941747u,2329300094u,686038586u,3468240685u,473715108u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 142309121u,3155208934u,2655173968u,1426461210u,1253492566u,2848333262u,2366551343u,2132894982u }  ;
              uint32_t vb[] = { 3362499285u,172969141u,445435847u,1692211347u,2569077277u,1768122433u,1036449931u,323645813u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 1816098025u,11010173u,673621419u,2511405706u,1983400372u,3862897534u,1874262304u,312060668u }  ;
              uint32_t vb[] = { 3507713220u,202903793u,1318285920u,997586398u,3088329905u,1609063172u,4087798908u,4281003629u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  12 );
        }
                {
              uint32_t va[] = { 2266187451u,1639702779u,450636760u,1409356569u,1327756721u,1445483954u,1684807079u,2899220875u }  ;
              uint32_t vb[] = { 3714708033u,1816279922u,236497618u,3452684905u,3371891668u,32956526u,1190834404u,2880171322u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 4232578243u,3085606665u,194321229u,4259232095u,2670255760u,1181891489u,1903926802u,2627481711u }  ;
              uint32_t vb[] = { 1370880777u,663588254u,2441904714u,4147647251u,1339067635u,1717011500u,2788223848u,966835619u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 1787491889u,2854622190u,2673478876u,2716424389u,3077739463u,2340024958u,2426907837u,1192302951u }  ;
              uint32_t vb[] = { 677672559u,2740207377u,869526993u,2727887665u,3137315176u,1584445350u,3745338260u,2874380948u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 3594924122u,1720104176u,3111152076u,1849052294u,2344014646u,2855267884u,643932902u,47293307u }  ;
              uint32_t vb[] = { 2317779377u,2651863920u,1899675062u,1911980220u,1254819188u,1461323505u,2780210111u,3950225629u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 3501285264u,1499992161u,509913684u,88704649u,2997489600u,3799796677u,3074138252u,1581242963u }  ;
              uint32_t vb[] = { 1282331496u,3900021003u,1057646577u,3827845888u,940439773u,1829628977u,1641833065u,2666227001u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -22 );
        }
                {
              uint32_t va[] = { 1939763907u,591981280u,445432352u,3454640140u,1892589004u,69752961u,1293799356u,1542219971u }  ;
              uint32_t vb[] = { 1846270938u,272319102u,2398539248u,4099937838u,2801335615u,1108616726u,443768384u,921498163u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 1138863059u,420819502u,939590593u,2258701744u,3582524802u,1815881314u,2484145859u,3126001450u }  ;
              uint32_t vb[] = { 1166993127u,2525652081u,1586297715u,291412028u,1218911883u,2980666333u,1336022890u,2096280905u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -32 );
        }
                {
              uint32_t va[] = { 1851225450u,3932402087u,2735324029u,931718494u,3330986347u,1237850945u,2068612602u,2320558829u }  ;
              uint32_t vb[] = { 2102018546u,415822617u,1632145734u,1343172361u,1955406334u,2705559135u,1579271888u,1780437235u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 209931361u,2874475908u,1052595392u,3406034651u,2581640641u,769913202u,1951926424u,2486430929u }  ;
              uint32_t vb[] = { 543266616u,4135827002u,4185616827u,1972599847u,1644175681u,3775660404u,1002403675u,2127956052u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -30 );
        }
                {
              uint32_t va[] = { 2463130829u,3819784689u,2803431318u,3959426379u,2106055184u,501796956u,279331376u,2688239494u }  ;
              uint32_t vb[] = { 1690207363u,506017983u,3816282023u,1137626138u,3305774988u,3168645980u,3687050468u,916364526u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 738475326u,3271396301u,662746992u,1578379091u,3557613023u,2671280412u,3368925184u,1721490106u }  ;
              uint32_t vb[] = { 3953102079u,3677700028u,1587989392u,2481185746u,2265792404u,2749651652u,495667344u,2871588179u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 785064147u,3103145093u,1990819026u,3780889547u,4059414753u,1286162237u,2541462832u,2946738932u }  ;
              uint32_t vb[] = { 591930041u,1436375035u,298489789u,3065319701u,955311755u,162338265u,614404115u,3036381902u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 3929222545u,56363561u,2038127404u,2583858381u,3290916937u,1010833371u,3377065876u,3750672170u }  ;
              uint32_t vb[] = { 408368988u,3194269824u,1867704230u,3710521625u,2170775942u,4129036025u,3658863732u,3927120089u }  ;
              size_t l = 256;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  12 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint32_t*, const uint32_t*, size_t), , dim=60") {
                {
              uint32_t va[] = { 477488800u,126393691u }  ;
              uint32_t vb[] = { 2923625333u,168807251u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 3787840991u,23154163u }  ;
              uint32_t vb[] = { 4265765640u,40541615u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 253569425u,246232446u }  ;
              uint32_t vb[] = { 3916399955u,163124109u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 1147658898u,233301805u }  ;
              uint32_t vb[] = { 4191793710u,200674954u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 596440929u,138240555u }  ;
              uint32_t vb[] = { 3018909989u,165667232u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  10 );
        }
                {
              uint32_t va[] = { 3512619343u,225649855u }  ;
              uint32_t vb[] = { 1323623030u,204572404u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 150024420u,232446869u }  ;
              uint32_t vb[] = { 2578751150u,212421256u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 2843692676u,81644991u }  ;
              uint32_t vb[] = { 2372157226u,144713609u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 1921773417u,223578874u }  ;
              uint32_t vb[] = { 3878977865u,80385901u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 4222958573u,247094101u }  ;
              uint32_t vb[] = { 2822207426u,185663750u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 2409184800u,160093674u }  ;
              uint32_t vb[] = { 3708589191u,124615237u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -14 );
        }
                {
              uint32_t va[] = { 1322917532u,58164975u }  ;
              uint32_t vb[] = { 1668397960u,203221902u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 850811636u,27906718u }  ;
              uint32_t vb[] = { 3891869374u,133002337u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 58839755u,189242859u }  ;
              uint32_t vb[] = { 4088197264u,25566352u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 2412448727u,189732659u }  ;
              uint32_t vb[] = { 3089880720u,75641002u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 1582423073u,194487568u }  ;
              uint32_t vb[] = { 428316106u,69557645u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -12 );
        }
                {
              uint32_t va[] = { 3173905244u,46662119u }  ;
              uint32_t vb[] = { 3052483655u,174753965u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  10 );
        }
                {
              uint32_t va[] = { 637192178u,20477484u }  ;
              uint32_t vb[] = { 1580212863u,129388567u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 1351479671u,242933572u }  ;
              uint32_t vb[] = { 1940525012u,140160755u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 3940382173u,261250738u }  ;
              uint32_t vb[] = { 1618842899u,217448028u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 2618964528u,64893124u }  ;
              uint32_t vb[] = { 2787844671u,7452243u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 388381527u,55747464u }  ;
              uint32_t vb[] = { 2923637473u,91661023u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 2699068643u,25532975u }  ;
              uint32_t vb[] = { 1747261343u,21011279u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  14 );
        }
                {
              uint32_t va[] = { 665894377u,13300367u }  ;
              uint32_t vb[] = { 2169483847u,245773681u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -20 );
        }
                {
              uint32_t va[] = { 2432959556u,199852284u }  ;
              uint32_t vb[] = { 3757870947u,16011457u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 596026905u,166334045u }  ;
              uint32_t vb[] = { 200811175u,23194835u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 2454204094u,178593280u }  ;
              uint32_t vb[] = { 3894488115u,264474936u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 3226363180u,7498875u }  ;
              uint32_t vb[] = { 1840644252u,250482700u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 3992035479u,119738092u }  ;
              uint32_t vb[] = { 2457931750u,13674275u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -18 );
        }
                {
              uint32_t va[] = { 2989932545u,115041783u }  ;
              uint32_t vb[] = { 3965637664u,176336033u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 3919429727u,120496317u }  ;
              uint32_t vb[] = { 569097053u,41429088u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 3029817106u,55336095u }  ;
              uint32_t vb[] = { 3644601042u,229553812u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 115212087u,146728851u }  ;
              uint32_t vb[] = { 402474223u,119925955u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 2087727837u,177800533u }  ;
              uint32_t vb[] = { 1188593695u,2647422u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 8665317u,61172836u }  ;
              uint32_t vb[] = { 2177734193u,24515107u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 2014305658u,60127449u }  ;
              uint32_t vb[] = { 3694189577u,175302709u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 3379238243u,244686u }  ;
              uint32_t vb[] = { 3635979690u,3839944u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  14 );
        }
                {
              uint32_t va[] = { 1681750928u,187333529u }  ;
              uint32_t vb[] = { 75931602u,206054031u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 1594654172u,107226341u }  ;
              uint32_t vb[] = { 2121051941u,69686809u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 1364606675u,20471917u }  ;
              uint32_t vb[] = { 3226008470u,5330667u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 1849415011u,213417728u }  ;
              uint32_t vb[] = { 1640461965u,12977018u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -16 );
        }
                {
              uint32_t va[] = { 2235201668u,109515321u }  ;
              uint32_t vb[] = { 3185021821u,235632734u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 1351199579u,212850097u }  ;
              uint32_t vb[] = { 1120176202u,91307369u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 3200665208u,246410395u }  ;
              uint32_t vb[] = { 3965672831u,166355622u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 3381271615u,4944904u }  ;
              uint32_t vb[] = { 2287086628u,81260044u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  16 );
        }
                {
              uint32_t va[] = { 1756819495u,260899064u }  ;
              uint32_t vb[] = { 3019341845u,249955045u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  10 );
        }
                {
              uint32_t va[] = { 3466479239u,45275650u }  ;
              uint32_t vb[] = { 2968178684u,134072724u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 3599142363u,12110386u }  ;
              uint32_t vb[] = { 2551653341u,172504476u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 3162743556u,143117507u }  ;
              uint32_t vb[] = { 2832539383u,189054476u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 2458634517u,32712912u }  ;
              uint32_t vb[] = { 1238986269u,110615465u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 1041277074u,82290276u }  ;
              uint32_t vb[] = { 467455862u,96622874u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 1566040503u,63535374u }  ;
              uint32_t vb[] = { 445085405u,145918637u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 205246589u,59035101u }  ;
              uint32_t vb[] = { 2959096939u,99168205u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 4259969981u,58258996u }  ;
              uint32_t vb[] = { 1368069650u,231385112u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 2678008312u,64975177u }  ;
              uint32_t vb[] = { 2174902443u,159302121u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 2438002110u,39873871u }  ;
              uint32_t vb[] = { 2424098656u,35478129u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 149977873u,86186834u }  ;
              uint32_t vb[] = { 1496150993u,109363609u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 4240493026u,32924562u }  ;
              uint32_t vb[] = { 544238942u,256614488u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 3527612818u,13079524u }  ;
              uint32_t vb[] = { 3681476260u,63325440u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 1232825906u,191206604u }  ;
              uint32_t vb[] = { 142772016u,244331470u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 1268803288u,68095360u }  ;
              uint32_t vb[] = { 2571181217u,13563213u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 3740516545u,54780325u }  ;
              uint32_t vb[] = { 108366362u,101468728u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 4074435241u,281959u }  ;
              uint32_t vb[] = { 1168927275u,39264606u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 3582326048u,247473952u }  ;
              uint32_t vb[] = { 3941896298u,68711686u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 2747051344u,128685334u }  ;
              uint32_t vb[] = { 145892758u,36621147u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 331528669u,32386370u }  ;
              uint32_t vb[] = { 2465333660u,149552662u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  14 );
        }
                {
              uint32_t va[] = { 1700141041u,201850225u }  ;
              uint32_t vb[] = { 1041264023u,207860271u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 337425221u,58361358u }  ;
              uint32_t vb[] = { 3662103294u,53587114u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 4281411863u,87005939u }  ;
              uint32_t vb[] = { 3134154853u,151164252u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 3368045295u,197029150u }  ;
              uint32_t vb[] = { 1567098448u,211543040u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 3486306778u,175527297u }  ;
              uint32_t vb[] = { 3632745664u,761326u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 858967962u,33535372u }  ;
              uint32_t vb[] = { 942401067u,107747712u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 1241161327u,183395038u }  ;
              uint32_t vb[] = { 2471189787u,109192127u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 1147736657u,211917080u }  ;
              uint32_t vb[] = { 2073835259u,254029719u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 3704256649u,217265939u }  ;
              uint32_t vb[] = { 2202199924u,139990508u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -14 );
        }
                {
              uint32_t va[] = { 1080829069u,187256076u }  ;
              uint32_t vb[] = { 2646664911u,186745477u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  10 );
        }
                {
              uint32_t va[] = { 2166056981u,265553555u }  ;
              uint32_t vb[] = { 1519988649u,36252907u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 1620911328u,48789433u }  ;
              uint32_t vb[] = { 3546967601u,118878838u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -14 );
        }
                {
              uint32_t va[] = { 3393155815u,222560207u }  ;
              uint32_t vb[] = { 2408401657u,208214819u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 1958063121u,92879862u }  ;
              uint32_t vb[] = { 150266244u,137589599u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 881816600u,52060666u }  ;
              uint32_t vb[] = { 3128859344u,134416095u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 2488512950u,172686343u }  ;
              uint32_t vb[] = { 362766670u,239794660u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  16 );
        }
                {
              uint32_t va[] = { 302933189u,214319151u }  ;
              uint32_t vb[] = { 213766805u,239239553u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 2529191325u,132829325u }  ;
              uint32_t vb[] = { 3699609597u,10164063u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 3378648741u,251750978u }  ;
              uint32_t vb[] = { 2570997825u,21302096u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  14 );
        }
                {
              uint32_t va[] = { 1828931925u,138319941u }  ;
              uint32_t vb[] = { 3280406101u,149196759u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  12 );
        }
                {
              uint32_t va[] = { 2012946975u,8075928u }  ;
              uint32_t vb[] = { 3455505235u,174851532u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 833648887u,85789879u }  ;
              uint32_t vb[] = { 4131171058u,22065957u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 2621594300u,24893778u }  ;
              uint32_t vb[] = { 233830909u,187900735u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 2961501907u,21695032u }  ;
              uint32_t vb[] = { 101290094u,250262602u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -16 );
        }
                {
              uint32_t va[] = { 4066747456u,147098155u }  ;
              uint32_t vb[] = { 2430409377u,196067722u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 2457344856u,141260697u }  ;
              uint32_t vb[] = { 3984115684u,193866860u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 2545397979u,99168643u }  ;
              uint32_t vb[] = { 3293105544u,209679165u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -12 );
        }
                {
              uint32_t va[] = { 1967269342u,24726658u }  ;
              uint32_t vb[] = { 1877674230u,37804102u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  10 );
        }
                {
              uint32_t va[] = { 3134183938u,30308004u }  ;
              uint32_t vb[] = { 2214088253u,15106050u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 3578663333u,9582027u }  ;
              uint32_t vb[] = { 359088207u,228137703u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 3590696739u,48931960u }  ;
              uint32_t vb[] = { 2488135365u,83059124u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  12 );
        }
                {
              uint32_t va[] = { 4215418446u,227528472u }  ;
              uint32_t vb[] = { 2097686130u,81996889u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  12 );
        }
                {
              uint32_t va[] = { 4034803909u,101655986u }  ;
              uint32_t vb[] = { 2796823928u,217584390u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 1568926719u,14792038u }  ;
              uint32_t vb[] = { 2618518683u,198460918u }  ;
              size_t l = 60;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  10 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint32_t*, const uint32_t*, size_t), , dim=59") {
                {
              uint32_t va[] = { 1932999322u,25004462u }  ;
              uint32_t vb[] = { 2710197360u,96962206u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -5 );
        }
                {
              uint32_t va[] = { 512844818u,131166139u }  ;
              uint32_t vb[] = { 1033420729u,74836978u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  7 );
        }
                {
              uint32_t va[] = { 800739696u,88608493u }  ;
              uint32_t vb[] = { 1432958804u,29364479u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  9 );
        }
                {
              uint32_t va[] = { 52013956u,113653221u }  ;
              uint32_t vb[] = { 3069197769u,21874120u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -7 );
        }
                {
              uint32_t va[] = { 1177417536u,79008101u }  ;
              uint32_t vb[] = { 1302164911u,574403u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  1 );
        }
                {
              uint32_t va[] = { 1798513227u,132319444u }  ;
              uint32_t vb[] = { 3053514924u,5207514u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -13 );
        }
                {
              uint32_t va[] = { 1747531821u,70539726u }  ;
              uint32_t vb[] = { 3569332787u,53119533u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -7 );
        }
                {
              uint32_t va[] = { 1512699396u,28567892u }  ;
              uint32_t vb[] = { 1099972519u,80203356u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  3 );
        }
                {
              uint32_t va[] = { 2435535448u,75954162u }  ;
              uint32_t vb[] = { 2222547784u,76564698u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  15 );
        }
                {
              uint32_t va[] = { 3391976505u,40316967u }  ;
              uint32_t vb[] = { 1709110215u,89354840u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -17 );
        }
                {
              uint32_t va[] = { 1506943073u,74896834u }  ;
              uint32_t vb[] = { 1827555280u,101251683u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -5 );
        }
                {
              uint32_t va[] = { 2762073658u,20216903u }  ;
              uint32_t vb[] = { 739189796u,19184342u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  15 );
        }
                {
              uint32_t va[] = { 1951565161u,22895071u }  ;
              uint32_t vb[] = { 1500850294u,109901346u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -17 );
        }
                {
              uint32_t va[] = { 1126525844u,5754205u }  ;
              uint32_t vb[] = { 930076310u,93984750u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -3 );
        }
                {
              uint32_t va[] = { 1787884991u,129181548u }  ;
              uint32_t vb[] = { 550668836u,52336800u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -9 );
        }
                {
              uint32_t va[] = { 1125566385u,39733134u }  ;
              uint32_t vb[] = { 435630232u,124870757u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -11 );
        }
                {
              uint32_t va[] = { 1501625844u,126997562u }  ;
              uint32_t vb[] = { 1042128550u,62995855u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -1 );
        }
                {
              uint32_t va[] = { 1146388103u,93908088u }  ;
              uint32_t vb[] = { 301897349u,124136835u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -3 );
        }
                {
              uint32_t va[] = { 2851696253u,81260164u }  ;
              uint32_t vb[] = { 3358144681u,90545675u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  5 );
        }
                {
              uint32_t va[] = { 43626482u,66078524u }  ;
              uint32_t vb[] = { 3214910599u,29669895u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -5 );
        }
                {
              uint32_t va[] = { 4201475923u,72835665u }  ;
              uint32_t vb[] = { 3329492857u,933378u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  7 );
        }
                {
              uint32_t va[] = { 4225565949u,21565651u }  ;
              uint32_t vb[] = { 1229956829u,70826825u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  7 );
        }
                {
              uint32_t va[] = { 2162371830u,83102826u }  ;
              uint32_t vb[] = { 1861098751u,39078153u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  11 );
        }
                {
              uint32_t va[] = { 3093394843u,128573628u }  ;
              uint32_t vb[] = { 634009712u,62658735u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -5 );
        }
                {
              uint32_t va[] = { 2522066718u,129481793u }  ;
              uint32_t vb[] = { 1203333850u,7436593u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -1 );
        }
                {
              uint32_t va[] = { 1651452301u,101383300u }  ;
              uint32_t vb[] = { 3048897032u,100690554u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -3 );
        }
                {
              uint32_t va[] = { 3900026415u,93397366u }  ;
              uint32_t vb[] = { 2188999868u,97479669u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  9 );
        }
                {
              uint32_t va[] = { 1570684129u,76645288u }  ;
              uint32_t vb[] = { 2273623196u,6729251u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -1 );
        }
                {
              uint32_t va[] = { 3143226114u,16837290u }  ;
              uint32_t vb[] = { 3203936994u,66460504u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  1 );
        }
                {
              uint32_t va[] = { 1878550476u,82620567u }  ;
              uint32_t vb[] = { 4294306060u,36033603u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  11 );
        }
                {
              uint32_t va[] = { 1392446599u,22482799u }  ;
              uint32_t vb[] = { 629837040u,133819193u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -7 );
        }
                {
              uint32_t va[] = { 676901905u,52728659u }  ;
              uint32_t vb[] = { 1846048032u,110974509u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  3 );
        }
                {
              uint32_t va[] = { 2318734282u,50576360u }  ;
              uint32_t vb[] = { 2758191098u,10964091u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -1 );
        }
                {
              uint32_t va[] = { 2096646885u,20611134u }  ;
              uint32_t vb[] = { 2048820341u,62938433u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -9 );
        }
                {
              uint32_t va[] = { 2717991681u,123916193u }  ;
              uint32_t vb[] = { 3916445552u,43028213u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  3 );
        }
                {
              uint32_t va[] = { 4286010875u,122344896u }  ;
              uint32_t vb[] = { 3217727638u,28867555u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -7 );
        }
                {
              uint32_t va[] = { 3737361101u,104911623u }  ;
              uint32_t vb[] = { 4001988553u,71729270u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  15 );
        }
                {
              uint32_t va[] = { 3991390064u,131572646u }  ;
              uint32_t vb[] = { 1705335322u,11127117u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -5 );
        }
                {
              uint32_t va[] = { 1347922575u,95392596u }  ;
              uint32_t vb[] = { 1158568648u,18071134u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  5 );
        }
                {
              uint32_t va[] = { 2877405495u,38518541u }  ;
              uint32_t vb[] = { 2361602218u,112170893u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  15 );
        }
                {
              uint32_t va[] = { 1177161580u,112716646u }  ;
              uint32_t vb[] = { 2070483501u,22386782u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  1 );
        }
                {
              uint32_t va[] = { 1714671394u,66534079u }  ;
              uint32_t vb[] = { 2822787168u,129350126u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  5 );
        }
                {
              uint32_t va[] = { 630034127u,40231646u }  ;
              uint32_t vb[] = { 600293605u,6056582u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  15 );
        }
                {
              uint32_t va[] = { 3558082289u,2385418u }  ;
              uint32_t vb[] = { 1188223473u,79049639u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  5 );
        }
                {
              uint32_t va[] = { 1834709587u,127334334u }  ;
              uint32_t vb[] = { 1149150212u,4994407u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -5 );
        }
                {
              uint32_t va[] = { 2047888385u,68457124u }  ;
              uint32_t vb[] = { 3924701358u,53489223u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -15 );
        }
                {
              uint32_t va[] = { 1702329550u,22752221u }  ;
              uint32_t vb[] = { 2895456692u,46586263u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -5 );
        }
                {
              uint32_t va[] = { 215247337u,16648098u }  ;
              uint32_t vb[] = { 4132167853u,86407049u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -1 );
        }
                {
              uint32_t va[] = { 2504234574u,17511900u }  ;
              uint32_t vb[] = { 1718734920u,19609302u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  21 );
        }
                {
              uint32_t va[] = { 2475551584u,52251623u }  ;
              uint32_t vb[] = { 3995061694u,23049556u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -7 );
        }
                {
              uint32_t va[] = { 4186265242u,122507417u }  ;
              uint32_t vb[] = { 3729361522u,125986892u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -3 );
        }
                {
              uint32_t va[] = { 2679581974u,25219721u }  ;
              uint32_t vb[] = { 1332714919u,91611120u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  1 );
        }
                {
              uint32_t va[] = { 139137979u,77172933u }  ;
              uint32_t vb[] = { 2328049163u,111772849u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  21 );
        }
                {
              uint32_t va[] = { 4121982648u,38361457u }  ;
              uint32_t vb[] = { 2784514252u,122268171u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  11 );
        }
                {
              uint32_t va[] = { 3439073302u,74464068u }  ;
              uint32_t vb[] = { 3715673583u,15569400u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -1 );
        }
                {
              uint32_t va[] = { 2893083592u,28609315u }  ;
              uint32_t vb[] = { 744739420u,48731924u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  13 );
        }
                {
              uint32_t va[] = { 3800480125u,59863163u }  ;
              uint32_t vb[] = { 3690058403u,74978651u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -9 );
        }
                {
              uint32_t va[] = { 92962985u,93953090u }  ;
              uint32_t vb[] = { 3564122746u,74010823u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -3 );
        }
                {
              uint32_t va[] = { 2724954009u,85140560u }  ;
              uint32_t vb[] = { 3444801820u,114907734u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -7 );
        }
                {
              uint32_t va[] = { 2648333320u,17687832u }  ;
              uint32_t vb[] = { 2988676097u,114176210u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -5 );
        }
                {
              uint32_t va[] = { 2943026497u,39078611u }  ;
              uint32_t vb[] = { 1440575571u,124453250u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -7 );
        }
                {
              uint32_t va[] = { 296941387u,37143825u }  ;
              uint32_t vb[] = { 3081443913u,7790673u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  19 );
        }
                {
              uint32_t va[] = { 524368675u,93971705u }  ;
              uint32_t vb[] = { 2105725345u,58322583u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  7 );
        }
                {
              uint32_t va[] = { 2317286198u,13770779u }  ;
              uint32_t vb[] = { 1293905917u,88651492u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -5 );
        }
                {
              uint32_t va[] = { 2308954267u,100739264u }  ;
              uint32_t vb[] = { 2521181389u,12991973u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -5 );
        }
                {
              uint32_t va[] = { 306884806u,70877102u }  ;
              uint32_t vb[] = { 1555830150u,124217628u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  1 );
        }
                {
              uint32_t va[] = { 1348231982u,85917978u }  ;
              uint32_t vb[] = { 2491374606u,39073068u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  5 );
        }
                {
              uint32_t va[] = { 2916763905u,36247652u }  ;
              uint32_t vb[] = { 671169490u,84030119u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -5 );
        }
                {
              uint32_t va[] = { 2636244731u,113862678u }  ;
              uint32_t vb[] = { 2650396486u,97275982u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  13 );
        }
                {
              uint32_t va[] = { 3393404239u,48316090u }  ;
              uint32_t vb[] = { 3114640775u,46109540u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -5 );
        }
                {
              uint32_t va[] = { 2820646081u,53088508u }  ;
              uint32_t vb[] = { 19639793u,101096274u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  9 );
        }
                {
              uint32_t va[] = { 4015411327u,4596304u }  ;
              uint32_t vb[] = { 3355111676u,118271389u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  5 );
        }
                {
              uint32_t va[] = { 1120210842u,42390596u }  ;
              uint32_t vb[] = { 3528162485u,84129202u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -3 );
        }
                {
              uint32_t va[] = { 650127090u,6420295u }  ;
              uint32_t vb[] = { 3761163497u,57714706u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -7 );
        }
                {
              uint32_t va[] = { 4145726687u,74373322u }  ;
              uint32_t vb[] = { 3722174116u,66913191u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -7 );
        }
                {
              uint32_t va[] = { 497466457u,22939728u }  ;
              uint32_t vb[] = { 2978315689u,88205503u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -1 );
        }
                {
              uint32_t va[] = { 534918987u,34708889u }  ;
              uint32_t vb[] = { 3485044390u,18196871u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  1 );
        }
                {
              uint32_t va[] = { 93358840u,36991419u }  ;
              uint32_t vb[] = { 900479907u,82370737u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  5 );
        }
                {
              uint32_t va[] = { 1263269892u,92119313u }  ;
              uint32_t vb[] = { 400600301u,52136654u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -9 );
        }
                {
              uint32_t va[] = { 2076121638u,54284858u }  ;
              uint32_t vb[] = { 329344481u,83717320u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -11 );
        }
                {
              uint32_t va[] = { 59981747u,106361642u }  ;
              uint32_t vb[] = { 1202983975u,84972743u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  1 );
        }
                {
              uint32_t va[] = { 1742160104u,26633425u }  ;
              uint32_t vb[] = { 3796212635u,123746899u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -5 );
        }
                {
              uint32_t va[] = { 2811558308u,101912426u }  ;
              uint32_t vb[] = { 2684450501u,114801538u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  7 );
        }
                {
              uint32_t va[] = { 525303965u,30355740u }  ;
              uint32_t vb[] = { 2416207761u,19370499u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -1 );
        }
                {
              uint32_t va[] = { 3289011264u,107555419u }  ;
              uint32_t vb[] = { 726388315u,16153727u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -3 );
        }
                {
              uint32_t va[] = { 3470510925u,129575343u }  ;
              uint32_t vb[] = { 2361045917u,81952595u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  1 );
        }
                {
              uint32_t va[] = { 2635394876u,9407948u }  ;
              uint32_t vb[] = { 2712032334u,48625058u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -5 );
        }
                {
              uint32_t va[] = { 3892131609u,102271863u }  ;
              uint32_t vb[] = { 315612992u,42792259u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -1 );
        }
                {
              uint32_t va[] = { 944765318u,130618682u }  ;
              uint32_t vb[] = { 2909639211u,96804329u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  9 );
        }
                {
              uint32_t va[] = { 551199066u,24051128u }  ;
              uint32_t vb[] = { 4223645715u,87318022u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -3 );
        }
                {
              uint32_t va[] = { 132772047u,29549149u }  ;
              uint32_t vb[] = { 3776722677u,28546602u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -7 );
        }
                {
              uint32_t va[] = { 74452669u,79658823u }  ;
              uint32_t vb[] = { 1168315409u,26244163u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  9 );
        }
                {
              uint32_t va[] = { 2740143119u,6296622u }  ;
              uint32_t vb[] = { 1580599016u,64645493u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -15 );
        }
                {
              uint32_t va[] = { 3703288029u,23456465u }  ;
              uint32_t vb[] = { 1809948341u,58780286u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  1 );
        }
                {
              uint32_t va[] = { 2605917473u,73504358u }  ;
              uint32_t vb[] = { 3734591413u,71448692u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  11 );
        }
                {
              uint32_t va[] = { 3925632587u,96407803u }  ;
              uint32_t vb[] = { 3352082909u,68505501u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -5 );
        }
                {
              uint32_t va[] = { 1953449961u,56901712u }  ;
              uint32_t vb[] = { 4198138075u,111759206u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -3 );
        }
                {
              uint32_t va[] = { 58302672u,8036670u }  ;
              uint32_t vb[] = { 243718637u,55804558u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -5 );
        }
                {
              uint32_t va[] = { 2560680311u,133040194u }  ;
              uint32_t vb[] = { 2269463608u,36780478u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -15 );
        }
                {
              uint32_t va[] = { 2370601021u,48563694u }  ;
              uint32_t vb[] = { 367325077u,115494872u }  ;
              size_t l = 59;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  7 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint32_t*, const uint32_t*, size_t), , dim=100") {
                {
              uint32_t va[] = { 850391183u,1043448035u,3818949369u,13u }  ;
              uint32_t vb[] = { 1994657798u,1435247203u,864876044u,9u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 907834159u,1093273059u,3956508452u,13u }  ;
              uint32_t vb[] = { 193382255u,3202737390u,1873346954u,14u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 2325963825u,1124768123u,1876153504u,7u }  ;
              uint32_t vb[] = { 11286841u,2536145177u,1847896939u,12u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 3311278187u,316190620u,692592511u,1u }  ;
              uint32_t vb[] = { 1452377058u,3100471580u,3446579261u,14u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 462766633u,3492383430u,3059659865u,4u }  ;
              uint32_t vb[] = { 3770926376u,962980187u,2290147317u,3u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -12 );
        }
                {
              uint32_t va[] = { 1207427136u,4006303319u,711773130u,15u }  ;
              uint32_t vb[] = { 1795256935u,3374588125u,3697792878u,10u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 2816220514u,3059262604u,934536631u,11u }  ;
              uint32_t vb[] = { 2452030786u,1559359795u,396493663u,11u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 3917796787u,1227117550u,467485660u,13u }  ;
              uint32_t vb[] = { 3475209102u,59231807u,2874835393u,0u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 315110356u,3888624554u,2636112834u,13u }  ;
              uint32_t vb[] = { 2844429015u,2063828102u,1598469275u,4u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 17003266u,393441995u,3640379652u,6u }  ;
              uint32_t vb[] = { 2180341430u,3613643069u,676644255u,5u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 211146418u,974161829u,2209046782u,0u }  ;
              uint32_t vb[] = { 1440139126u,190797790u,3656769092u,0u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 4288956673u,1463208620u,417383191u,11u }  ;
              uint32_t vb[] = { 603562853u,1960557262u,1665081166u,6u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -16 );
        }
                {
              uint32_t va[] = { 2832906899u,1066919103u,1421384500u,11u }  ;
              uint32_t vb[] = { 3338963545u,2106121525u,2889040326u,12u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 2609252632u,2926693386u,3053161725u,10u }  ;
              uint32_t vb[] = { 3166484542u,3203410508u,9612389u,0u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  18 );
        }
                {
              uint32_t va[] = { 1845067232u,4729883u,2087471320u,13u }  ;
              uint32_t vb[] = { 3125142575u,3850395209u,3979826199u,15u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 2419999823u,4096224175u,3907955636u,11u }  ;
              uint32_t vb[] = { 3936717325u,419801263u,2787570319u,9u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 1221740689u,4200078175u,1366799702u,7u }  ;
              uint32_t vb[] = { 4119024582u,1131652006u,887653342u,14u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 2106916436u,3951501374u,3970183219u,10u }  ;
              uint32_t vb[] = { 3629292063u,163417504u,2297510935u,0u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 3690332478u,1479546533u,1544862153u,14u }  ;
              uint32_t vb[] = { 2884698436u,1668066360u,870682632u,4u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 1792317341u,410248476u,1634749706u,0u }  ;
              uint32_t vb[] = { 2876011050u,3313115430u,619409132u,5u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 3530059827u,738463490u,4013215084u,11u }  ;
              uint32_t vb[] = { 3248164780u,502619482u,1046828040u,4u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 1253279499u,3522688910u,2622586277u,0u }  ;
              uint32_t vb[] = { 3312643241u,810110164u,88768264u,6u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 3765104322u,1549483017u,4241091451u,8u }  ;
              uint32_t vb[] = { 4052575280u,3147752454u,2808721930u,5u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 2981852532u,1763192784u,3440503210u,7u }  ;
              uint32_t vb[] = { 4260634297u,183915257u,1538833787u,11u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 2302200221u,2461529086u,2211581789u,10u }  ;
              uint32_t vb[] = { 2793568547u,1237342650u,1105614385u,8u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 2196659196u,1313706319u,1155391119u,2u }  ;
              uint32_t vb[] = { 611531549u,1194603931u,550803651u,6u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  18 );
        }
                {
              uint32_t va[] = { 1689350550u,1692019852u,1290214801u,9u }  ;
              uint32_t vb[] = { 1350865893u,1791118062u,969063049u,10u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  10 );
        }
                {
              uint32_t va[] = { 313902006u,704966137u,488927315u,15u }  ;
              uint32_t vb[] = { 2484743821u,3425511847u,201558513u,7u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 489129139u,2715781968u,3531992808u,14u }  ;
              uint32_t vb[] = { 2138418359u,817939357u,3545764033u,11u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  14 );
        }
                {
              uint32_t va[] = { 4006362621u,610065393u,3527030412u,9u }  ;
              uint32_t vb[] = { 3208581161u,3968470062u,2720876061u,13u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 596926370u,2139330938u,1562747053u,9u }  ;
              uint32_t vb[] = { 773796713u,298703277u,1568536055u,8u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  12 );
        }
                {
              uint32_t va[] = { 2660295851u,2755676645u,4074833897u,9u }  ;
              uint32_t vb[] = { 3547340082u,2864251165u,4272399235u,5u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 95439500u,1453223714u,3581616806u,7u }  ;
              uint32_t vb[] = { 3563034145u,2575427386u,527486355u,10u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -14 );
        }
                {
              uint32_t va[] = { 4106007568u,2382221269u,3284008333u,14u }  ;
              uint32_t vb[] = { 161274406u,2175683733u,3483645157u,2u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  18 );
        }
                {
              uint32_t va[] = { 395728997u,2511888177u,580766568u,10u }  ;
              uint32_t vb[] = { 1372820152u,3310979133u,3363105563u,8u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 1405550681u,4252269222u,728478655u,9u }  ;
              uint32_t vb[] = { 3242181440u,1341056030u,3199126314u,5u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 3304802240u,3965189447u,3463215987u,5u }  ;
              uint32_t vb[] = { 3866546137u,3726687787u,3889640603u,4u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 3954426406u,1460651322u,3346498971u,4u }  ;
              uint32_t vb[] = { 1763790734u,762739769u,2491264928u,9u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 2637609202u,1489630387u,830116144u,12u }  ;
              uint32_t vb[] = { 826621200u,1646348007u,416933789u,2u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 4291900108u,1097985089u,2639240558u,4u }  ;
              uint32_t vb[] = { 3795344798u,572227584u,169271911u,11u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 1061192552u,2525644839u,4034280290u,1u }  ;
              uint32_t vb[] = { 740825790u,3376468179u,41156430u,9u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 3711600712u,1963749022u,2832411655u,6u }  ;
              uint32_t vb[] = { 2720052530u,3273658664u,3790561063u,7u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 4098971190u,3390626258u,1044960683u,1u }  ;
              uint32_t vb[] = { 1386427102u,2250997775u,2893709293u,15u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 1002486999u,3629093181u,4051068986u,3u }  ;
              uint32_t vb[] = { 393147847u,456303099u,3289869112u,8u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  18 );
        }
                {
              uint32_t va[] = { 3440073955u,561595140u,1680753152u,5u }  ;
              uint32_t vb[] = { 3287321545u,582948056u,454404857u,3u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -18 );
        }
                {
              uint32_t va[] = { 2473169589u,3109580568u,2005553959u,4u }  ;
              uint32_t vb[] = { 2140487457u,531773900u,1227825029u,1u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 2235824595u,3176903554u,355209729u,4u }  ;
              uint32_t vb[] = { 2817849862u,112601372u,2724766050u,8u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 96491230u,189172021u,2358990059u,3u }  ;
              uint32_t vb[] = { 3471921292u,362471604u,11178633u,2u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  16 );
        }
                {
              uint32_t va[] = { 2157762949u,2831961268u,36597817u,15u }  ;
              uint32_t vb[] = { 547807237u,3501760820u,1743268626u,15u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  14 );
        }
                {
              uint32_t va[] = { 2111437098u,349744914u,2581263603u,12u }  ;
              uint32_t vb[] = { 2523161132u,123852985u,756851333u,0u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -14 );
        }
                {
              uint32_t va[] = { 2359096340u,1308442690u,3941816454u,4u }  ;
              uint32_t vb[] = { 1555603771u,2983402114u,132846433u,6u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 3988985965u,1158981099u,2390564808u,11u }  ;
              uint32_t vb[] = { 893973668u,532434424u,53549722u,15u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  12 );
        }
                {
              uint32_t va[] = { 1816375792u,715273978u,4280802165u,15u }  ;
              uint32_t vb[] = { 3411483653u,3493671613u,749953381u,9u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 1985050600u,4209646381u,3065827948u,10u }  ;
              uint32_t vb[] = { 4019790746u,1899071542u,2498196193u,15u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 2350313307u,803313282u,3179456535u,10u }  ;
              uint32_t vb[] = { 3291199807u,1899540135u,1472835374u,4u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 3425009698u,3891194523u,3840859236u,8u }  ;
              uint32_t vb[] = { 982197267u,1454440514u,4261024567u,13u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 1381731254u,1934906582u,2315126263u,6u }  ;
              uint32_t vb[] = { 929745976u,2800687188u,3216314946u,1u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 1967966019u,3607168231u,388168877u,15u }  ;
              uint32_t vb[] = { 101883601u,1046719221u,3504499066u,2u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 2861649137u,543352364u,3668449565u,4u }  ;
              uint32_t vb[] = { 1708565589u,1023326450u,2804943969u,0u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 3779110396u,4253123929u,3021782321u,3u }  ;
              uint32_t vb[] = { 615793019u,1275345717u,3157759224u,10u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 1337913649u,394556446u,717483477u,4u }  ;
              uint32_t vb[] = { 869332398u,1255207404u,1426490037u,8u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 3058778807u,2351536292u,679954574u,2u }  ;
              uint32_t vb[] = { 3365503602u,1355307250u,2229696560u,5u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 2169190376u,199533135u,2289980365u,11u }  ;
              uint32_t vb[] = { 1100642864u,2977623746u,3650127208u,5u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 2055502285u,2419806482u,3213843519u,7u }  ;
              uint32_t vb[] = { 3663638685u,814604675u,672955425u,1u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  16 );
        }
                {
              uint32_t va[] = { 2337371969u,100371662u,2759715398u,8u }  ;
              uint32_t vb[] = { 3314121672u,867684213u,1388313401u,14u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 3750932325u,162465180u,2227215215u,15u }  ;
              uint32_t vb[] = { 3626639114u,2818553977u,3567245616u,15u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 3516829880u,2029480993u,1274016363u,7u }  ;
              uint32_t vb[] = { 638312395u,3122423431u,868501489u,2u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 1374740824u,3909663733u,1177997030u,11u }  ;
              uint32_t vb[] = { 1352331964u,2592145483u,3713684935u,15u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 1151127755u,3534592123u,1873802109u,0u }  ;
              uint32_t vb[] = { 3515161526u,1116563088u,311865287u,3u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 2740669837u,4108265438u,3556740378u,3u }  ;
              uint32_t vb[] = { 1928093485u,3028036373u,1375788938u,12u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  10 );
        }
                {
              uint32_t va[] = { 3989867766u,1951236473u,1139527153u,8u }  ;
              uint32_t vb[] = { 1109614119u,4114551133u,4152036271u,10u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 3923638621u,1290281121u,2406022664u,10u }  ;
              uint32_t vb[] = { 4268006379u,2144441745u,4055284246u,0u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 810890377u,2115367398u,1425756495u,6u }  ;
              uint32_t vb[] = { 1054249485u,2997071228u,1131028723u,15u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 164394341u,2002249239u,2515993633u,10u }  ;
              uint32_t vb[] = { 2078028451u,1048710427u,324725897u,11u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  10 );
        }
                {
              uint32_t va[] = { 4170163436u,3991375348u,3895876624u,2u }  ;
              uint32_t vb[] = { 2014133747u,475466266u,2417004705u,10u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 4254840154u,1920992880u,302455046u,11u }  ;
              uint32_t vb[] = { 3278641699u,8528134u,4191444608u,8u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -14 );
        }
                {
              uint32_t va[] = { 399980831u,4287995686u,389543280u,7u }  ;
              uint32_t vb[] = { 2432637752u,3755061536u,1035435051u,6u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  32 );
        }
                {
              uint32_t va[] = { 2250943010u,655939634u,3159658620u,6u }  ;
              uint32_t vb[] = { 2884458580u,673144907u,1045967243u,5u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 717522184u,3913076312u,4135273405u,10u }  ;
              uint32_t vb[] = { 1515100212u,1083862766u,950000115u,5u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 374815201u,3907622000u,3578381224u,5u }  ;
              uint32_t vb[] = { 8636816u,1429882284u,3081107936u,3u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 4017081392u,2568976055u,3657735928u,14u }  ;
              uint32_t vb[] = { 3811890111u,3212095169u,1922840590u,2u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 2572486204u,3361812544u,810105298u,4u }  ;
              uint32_t vb[] = { 2373734331u,3942430808u,2644569851u,0u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  14 );
        }
                {
              uint32_t va[] = { 3145858741u,1422093635u,1945647488u,8u }  ;
              uint32_t vb[] = { 4173575897u,3718774363u,47161158u,5u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  14 );
        }
                {
              uint32_t va[] = { 210122752u,2700883946u,3330806007u,5u }  ;
              uint32_t vb[] = { 243849637u,2024993998u,3941599809u,0u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  14 );
        }
                {
              uint32_t va[] = { 1401657484u,1236517021u,4189494244u,7u }  ;
              uint32_t vb[] = { 1410395203u,2284697212u,20175477u,14u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 292969035u,3734195600u,698493441u,7u }  ;
              uint32_t vb[] = { 1136291543u,2174732855u,2185503745u,14u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 1589111089u,1550243138u,2585377830u,5u }  ;
              uint32_t vb[] = { 2034300597u,846987803u,363786282u,15u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 4200694994u,2217461356u,4096525007u,3u }  ;
              uint32_t vb[] = { 514134950u,195237069u,2519763259u,12u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 1362786990u,2019216712u,3488657420u,8u }  ;
              uint32_t vb[] = { 3080457101u,269044125u,1150545075u,10u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 2227911329u,2206801580u,3419328131u,3u }  ;
              uint32_t vb[] = { 2516175970u,757291348u,999827933u,1u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 438658713u,1233063020u,3861061785u,8u }  ;
              uint32_t vb[] = { 661172515u,2087605236u,78164113u,3u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 2446509325u,233208526u,1661408960u,9u }  ;
              uint32_t vb[] = { 1198868747u,2580458340u,4279394960u,14u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  12 );
        }
                {
              uint32_t va[] = { 3396917397u,1107470702u,3136165u,14u }  ;
              uint32_t vb[] = { 3027239493u,3621468187u,2618487823u,5u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 350869853u,3606266048u,2277490007u,10u }  ;
              uint32_t vb[] = { 221071510u,360799221u,2144802942u,11u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 615413633u,1801006324u,2076151269u,7u }  ;
              uint32_t vb[] = { 260299972u,3706591008u,3138730264u,14u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -12 );
        }
                {
              uint32_t va[] = { 918160274u,789075439u,3652121804u,6u }  ;
              uint32_t vb[] = { 3724022300u,4066681026u,3489875976u,7u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 3277647433u,294564580u,2009130421u,6u }  ;
              uint32_t vb[] = { 2511131547u,2811923930u,2113357846u,2u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 1931135775u,1018843390u,3675796315u,2u }  ;
              uint32_t vb[] = { 1267813945u,3165170866u,525768642u,9u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  16 );
        }
                {
              uint32_t va[] = { 838056889u,1219421920u,1561359990u,9u }  ;
              uint32_t vb[] = { 226642420u,2886672240u,4112154233u,14u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  10 );
        }
                {
              uint32_t va[] = { 1410039162u,3178875085u,480207495u,1u }  ;
              uint32_t vb[] = { 1537775408u,3700291065u,108231935u,0u }  ;
              size_t l = 100;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint32_t*, const uint32_t*, size_t), , dim=119") {
                {
              uint32_t va[] = { 4043600883u,953782728u,432880815u,2945463u }  ;
              uint32_t vb[] = { 531301955u,3502493579u,2466468974u,1889861u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -1 );
        }
                {
              uint32_t va[] = { 3952173571u,1263564089u,2347365804u,7564987u }  ;
              uint32_t vb[] = { 2585096977u,94254506u,3577573686u,1925052u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  19 );
        }
                {
              uint32_t va[] = { 3525662283u,1433777335u,2819725526u,6794326u }  ;
              uint32_t vb[] = { 2398086526u,1600888828u,1883682449u,4041200u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  3 );
        }
                {
              uint32_t va[] = { 1760265569u,2842383278u,251541749u,5310495u }  ;
              uint32_t vb[] = { 4257680868u,2474600100u,1470945708u,5258909u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  27 );
        }
                {
              uint32_t va[] = { 3478617696u,138974689u,3196780496u,3043674u }  ;
              uint32_t vb[] = { 4102810397u,3168471415u,671745594u,6074103u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -17 );
        }
                {
              uint32_t va[] = { 606854701u,1334004180u,840254192u,6136349u }  ;
              uint32_t vb[] = { 1573035511u,3447777161u,865227875u,3452149u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  9 );
        }
                {
              uint32_t va[] = { 2487258252u,2567566861u,2882624049u,1699395u }  ;
              uint32_t vb[] = { 348140029u,2515338095u,1470192405u,1975339u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  21 );
        }
                {
              uint32_t va[] = { 2571341755u,607047006u,194703478u,2715587u }  ;
              uint32_t vb[] = { 1842985936u,3576246434u,1937775319u,250458u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -13 );
        }
                {
              uint32_t va[] = { 4159927835u,1714197944u,954851406u,6081075u }  ;
              uint32_t vb[] = { 3649572833u,2801073469u,2209994138u,4073454u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -9 );
        }
                {
              uint32_t va[] = { 4282411245u,2168046706u,321095766u,5163861u }  ;
              uint32_t vb[] = { 1466364411u,3478939423u,2528905296u,4992391u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  15 );
        }
                {
              uint32_t va[] = { 297285682u,1756900384u,623442028u,5547637u }  ;
              uint32_t vb[] = { 159804830u,779200212u,90796656u,7387733u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  29 );
        }
                {
              uint32_t va[] = { 924285931u,3930749608u,2735210839u,8302074u }  ;
              uint32_t vb[] = { 3651959896u,3071707020u,2223636675u,575180u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -5 );
        }
                {
              uint32_t va[] = { 2235162572u,2327909032u,3547525304u,6573610u }  ;
              uint32_t vb[] = { 1859883829u,1657768730u,1913013807u,3304930u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -3 );
        }
                {
              uint32_t va[] = { 3257343635u,701254463u,3836689260u,1981191u }  ;
              uint32_t vb[] = { 3556884060u,3488646490u,1320540769u,122040u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -9 );
        }
                {
              uint32_t va[] = { 3510999780u,1593797972u,1668422134u,3048826u }  ;
              uint32_t vb[] = { 1077905153u,844048082u,3986111601u,160638u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  5 );
        }
                {
              uint32_t va[] = { 3121492647u,1974115081u,395088765u,464369u }  ;
              uint32_t vb[] = { 1482032143u,408979740u,4002714205u,3695975u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  17 );
        }
                {
              uint32_t va[] = { 2056073975u,1967180010u,995647909u,1345721u }  ;
              uint32_t vb[] = { 836641071u,726321480u,2146984345u,4064811u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  31 );
        }
                {
              uint32_t va[] = { 738713573u,992825808u,582387188u,1557358u }  ;
              uint32_t vb[] = { 2147167100u,567448375u,2979826248u,1778665u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -9 );
        }
                {
              uint32_t va[] = { 2313696850u,1441612352u,3465943952u,6522530u }  ;
              uint32_t vb[] = { 334286908u,514835669u,3953853838u,7715262u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  7 );
        }
                {
              uint32_t va[] = { 507857866u,1869787908u,2610801354u,6529611u }  ;
              uint32_t vb[] = { 3655510331u,1386811608u,1180795495u,3536915u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -19 );
        }
                {
              uint32_t va[] = { 1766157923u,4115126721u,3060064266u,5902098u }  ;
              uint32_t vb[] = { 3123426901u,4158143156u,2303652504u,8107359u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -1 );
        }
                {
              uint32_t va[] = { 1033819958u,3084130313u,1830699995u,6761078u }  ;
              uint32_t vb[] = { 1497938639u,3028356241u,360771370u,4731353u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -7 );
        }
                {
              uint32_t va[] = { 1729813620u,164569207u,292293811u,7232084u }  ;
              uint32_t vb[] = { 303869681u,1947585814u,1311030192u,8165733u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  1 );
        }
                {
              uint32_t va[] = { 4204208760u,405951576u,3177198020u,5666116u }  ;
              uint32_t vb[] = { 1761454655u,1627791657u,4110216315u,4297162u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -5 );
        }
                {
              uint32_t va[] = { 2926713403u,3574867156u,3905055572u,977245u }  ;
              uint32_t vb[] = { 2536981022u,638189675u,2720872116u,1659421u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -3 );
        }
                {
              uint32_t va[] = { 3005378271u,258140837u,2900221798u,6066793u }  ;
              uint32_t vb[] = { 3127212136u,1323407046u,4259890846u,1372388u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  13 );
        }
                {
              uint32_t va[] = { 2161056104u,3964461067u,1296722033u,7703095u }  ;
              uint32_t vb[] = { 1489512663u,3831155910u,107390992u,3468516u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  15 );
        }
                {
              uint32_t va[] = { 1275964299u,934298693u,966067833u,5683522u }  ;
              uint32_t vb[] = { 781761805u,4727420u,2117936086u,5598499u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  7 );
        }
                {
              uint32_t va[] = { 2905356879u,485376434u,1347834133u,328441u }  ;
              uint32_t vb[] = { 1459162405u,2457308064u,2915059198u,2397513u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -9 );
        }
                {
              uint32_t va[] = { 1169547052u,1385608182u,926857886u,2693302u }  ;
              uint32_t vb[] = { 739018911u,2007546152u,469023259u,4659409u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  9 );
        }
                {
              uint32_t va[] = { 2852982275u,157142910u,367016114u,2808838u }  ;
              uint32_t vb[] = { 105497962u,768886721u,2581921289u,6492281u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  5 );
        }
                {
              uint32_t va[] = { 1282314424u,2595916711u,3250853598u,6449277u }  ;
              uint32_t vb[] = { 3685356957u,4055773096u,2970591721u,5676364u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  3 );
        }
                {
              uint32_t va[] = { 3332421503u,824507446u,829237323u,197937u }  ;
              uint32_t vb[] = { 2174607830u,3739733235u,219453398u,8290393u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -15 );
        }
                {
              uint32_t va[] = { 398651718u,1274975950u,1468258949u,6763346u }  ;
              uint32_t vb[] = { 1772264937u,1044569342u,387393955u,2612145u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  7 );
        }
                {
              uint32_t va[] = { 1572801374u,2222998171u,3348092899u,5208825u }  ;
              uint32_t vb[] = { 3192292119u,2086224240u,4178245144u,2586786u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -1 );
        }
                {
              uint32_t va[] = { 1165044454u,3424266466u,579828705u,813652u }  ;
              uint32_t vb[] = { 1461789468u,3094603181u,1874222665u,2034121u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  7 );
        }
                {
              uint32_t va[] = { 2323672956u,2137009511u,1133199036u,5225157u }  ;
              uint32_t vb[] = { 229359842u,529686153u,526981396u,4330683u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -13 );
        }
                {
              uint32_t va[] = { 2861151438u,1864004771u,1232470125u,6385329u }  ;
              uint32_t vb[] = { 1194984176u,361236290u,3010304318u,7912112u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -9 );
        }
                {
              uint32_t va[] = { 3425306035u,1846544685u,111587561u,6734526u }  ;
              uint32_t vb[] = { 2393664418u,1135860878u,1758245887u,6303572u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  7 );
        }
                {
              uint32_t va[] = { 3141524522u,2009801060u,3242600064u,1152472u }  ;
              uint32_t vb[] = { 2031482387u,3983510005u,985828036u,7744196u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  7 );
        }
                {
              uint32_t va[] = { 3268351153u,475856360u,339144859u,7558182u }  ;
              uint32_t vb[] = { 4060721869u,3872144546u,3583906937u,154158u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  17 );
        }
                {
              uint32_t va[] = { 176364984u,850624028u,179386836u,5351158u }  ;
              uint32_t vb[] = { 3127550447u,3728788621u,3625677925u,3845268u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -5 );
        }
                {
              uint32_t va[] = { 1754447106u,2412744623u,3716921078u,2874853u }  ;
              uint32_t vb[] = { 3461894613u,3343529962u,398072180u,6305211u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  1 );
        }
                {
              uint32_t va[] = { 3463684564u,3720819008u,928393790u,3923753u }  ;
              uint32_t vb[] = { 579773910u,408962462u,1869272443u,657034u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -7 );
        }
                {
              uint32_t va[] = { 3014814056u,905198006u,3770040368u,7020560u }  ;
              uint32_t vb[] = { 2624945591u,3504917089u,247473902u,6386072u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -15 );
        }
                {
              uint32_t va[] = { 2610584989u,2113152613u,2484047532u,3011169u }  ;
              uint32_t vb[] = { 925002021u,1717720748u,2060662774u,5776336u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -7 );
        }
                {
              uint32_t va[] = { 2406347908u,714542466u,3073177616u,2360098u }  ;
              uint32_t vb[] = { 826288058u,4090022745u,916245659u,3566694u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -7 );
        }
                {
              uint32_t va[] = { 2768809357u,700651471u,3234304380u,3045365u }  ;
              uint32_t vb[] = { 2144856542u,2139539597u,61789427u,7127920u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  1 );
        }
                {
              uint32_t va[] = { 2479287574u,4179769670u,329182586u,7553489u }  ;
              uint32_t vb[] = { 2064146235u,3789673263u,92962164u,7778510u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  11 );
        }
                {
              uint32_t va[] = { 667484909u,460236852u,1431858818u,8006942u }  ;
              uint32_t vb[] = { 1813239925u,2525292926u,2812887159u,739390u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -1 );
        }
                {
              uint32_t va[] = { 2022139046u,1907782260u,950843844u,5040132u }  ;
              uint32_t vb[] = { 3242258895u,2744666354u,2435250832u,7919841u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  7 );
        }
                {
              uint32_t va[] = { 3970030740u,4157321959u,462373320u,8044463u }  ;
              uint32_t vb[] = { 1712460092u,1063811828u,346965545u,3429201u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  3 );
        }
                {
              uint32_t va[] = { 1448683210u,3042507899u,247001308u,5699110u }  ;
              uint32_t vb[] = { 2167071788u,3708935417u,1649404534u,7277374u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  1 );
        }
                {
              uint32_t va[] = { 1102601651u,3176543354u,1458271569u,4970568u }  ;
              uint32_t vb[] = { 581568560u,2210939403u,183046740u,6277706u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  21 );
        }
                {
              uint32_t va[] = { 2938547268u,3112746383u,2452740738u,3705659u }  ;
              uint32_t vb[] = { 273805818u,3305863942u,3147439780u,724898u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -7 );
        }
                {
              uint32_t va[] = { 759835742u,3779968420u,3021611704u,4310083u }  ;
              uint32_t vb[] = { 2109262346u,2013792654u,1145859220u,6032247u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  5 );
        }
                {
              uint32_t va[] = { 1378426803u,3143858294u,2552433154u,7660486u }  ;
              uint32_t vb[] = { 897422385u,2706083411u,2305706729u,6582722u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  27 );
        }
                {
              uint32_t va[] = { 3411628298u,4121576334u,2874270714u,7140652u }  ;
              uint32_t vb[] = { 3978894857u,2614175196u,1001849828u,4316898u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  3 );
        }
                {
              uint32_t va[] = { 4293001465u,1174940108u,4051560584u,684929u }  ;
              uint32_t vb[] = { 644511572u,3935703276u,1208892348u,1000494u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -7 );
        }
                {
              uint32_t va[] = { 1787689351u,1183849900u,1975210402u,1841310u }  ;
              uint32_t vb[] = { 1856256889u,516982631u,3865906376u,2750926u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  9 );
        }
                {
              uint32_t va[] = { 1063364701u,2467780698u,901127723u,2915539u }  ;
              uint32_t vb[] = { 3705922964u,4280967377u,1872960188u,2880761u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  15 );
        }
                {
              uint32_t va[] = { 2650430327u,637147497u,474297136u,6368333u }  ;
              uint32_t vb[] = { 878242981u,3046898247u,1828062878u,5243569u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -3 );
        }
                {
              uint32_t va[] = { 953842345u,929605673u,1557870971u,2629384u }  ;
              uint32_t vb[] = { 1268817965u,1525658218u,2245291635u,5645056u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  3 );
        }
                {
              uint32_t va[] = { 2436367531u,846395583u,1962209928u,1395609u }  ;
              uint32_t vb[] = { 531471420u,568668285u,264194444u,1043638u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -1 );
        }
                {
              uint32_t va[] = { 4079112600u,1699327199u,4049151261u,3279600u }  ;
              uint32_t vb[] = { 2845749331u,1971556139u,3199431595u,291686u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -7 );
        }
                {
              uint32_t va[] = { 3441989183u,1710484307u,940178615u,5883005u }  ;
              uint32_t vb[] = { 2137338958u,3058614256u,4206801023u,1377073u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  1 );
        }
                {
              uint32_t va[] = { 710953768u,2414473256u,3893585849u,641470u }  ;
              uint32_t vb[] = { 319922600u,4122942796u,1270698630u,2710607u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -3 );
        }
                {
              uint32_t va[] = { 1106835714u,4237807741u,3364669655u,5645266u }  ;
              uint32_t vb[] = { 3803896704u,775441761u,145213626u,2072272u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  21 );
        }
                {
              uint32_t va[] = { 3214054904u,1317637024u,3384390379u,1315629u }  ;
              uint32_t vb[] = { 2209418948u,1224165060u,1221448155u,7887643u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  11 );
        }
                {
              uint32_t va[] = { 3371402041u,4178056998u,790313739u,5526492u }  ;
              uint32_t vb[] = { 3015211362u,3190117126u,2873558385u,6834474u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -3 );
        }
                {
              uint32_t va[] = { 1539061025u,3646012011u,1729985204u,3477295u }  ;
              uint32_t vb[] = { 447526664u,2360567937u,933933670u,1169191u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  25 );
        }
                {
              uint32_t va[] = { 1355845045u,2592947896u,134794323u,1889625u }  ;
              uint32_t vb[] = { 679228388u,1152323539u,2197620813u,3768936u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -1 );
        }
                {
              uint32_t va[] = { 1963337329u,4195748104u,1398496808u,7940940u }  ;
              uint32_t vb[] = { 1260563061u,1856656566u,4195733415u,5519671u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  3 );
        }
                {
              uint32_t va[] = { 2934214914u,3788319748u,309326386u,4689491u }  ;
              uint32_t vb[] = { 3544979163u,616256379u,4237984904u,2515592u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -35 );
        }
                {
              uint32_t va[] = { 1438798084u,1067031239u,2324895543u,1499u }  ;
              uint32_t vb[] = { 597597347u,3056927411u,1256511556u,4062342u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -17 );
        }
                {
              uint32_t va[] = { 1019356762u,1713528759u,2614936702u,6391958u }  ;
              uint32_t vb[] = { 2356726415u,2349850134u,1779852420u,5341590u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  13 );
        }
                {
              uint32_t va[] = { 805687788u,2651496695u,80568446u,2990943u }  ;
              uint32_t vb[] = { 1136336879u,1274734264u,2489484933u,1962542u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -9 );
        }
                {
              uint32_t va[] = { 3296125758u,2532543067u,881772743u,8301251u }  ;
              uint32_t vb[] = { 1828340533u,2643967767u,1595077266u,3270627u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  17 );
        }
                {
              uint32_t va[] = { 159226965u,3830522181u,1451273909u,1230830u }  ;
              uint32_t vb[] = { 3267050693u,895357073u,984482870u,6394803u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  7 );
        }
                {
              uint32_t va[] = { 3333901699u,2091325033u,2469799266u,128650u }  ;
              uint32_t vb[] = { 433268198u,4054023025u,1212857848u,6693314u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -1 );
        }
                {
              uint32_t va[] = { 1723511398u,1332815628u,1987753146u,7483670u }  ;
              uint32_t vb[] = { 2163484724u,3823193302u,2908887607u,7488292u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  7 );
        }
                {
              uint32_t va[] = { 3081866294u,2318923801u,3083949164u,4418312u }  ;
              uint32_t vb[] = { 1985745273u,3980629740u,2945761238u,5542985u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -19 );
        }
                {
              uint32_t va[] = { 85826276u,1736273168u,2336548746u,1643507u }  ;
              uint32_t vb[] = { 1869967527u,2324964003u,3005745186u,3852143u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -9 );
        }
                {
              uint32_t va[] = { 2700731656u,937723146u,833090642u,8309963u }  ;
              uint32_t vb[] = { 1016962107u,1818908131u,2170222541u,2833392u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -15 );
        }
                {
              uint32_t va[] = { 401285134u,3483809391u,1343398178u,6074682u }  ;
              uint32_t vb[] = { 3933719115u,3954183437u,106996274u,6456549u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -1 );
        }
                {
              uint32_t va[] = { 79186835u,152804746u,2292826262u,6304540u }  ;
              uint32_t vb[] = { 3014436146u,317301805u,3226909619u,1434167u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -15 );
        }
                {
              uint32_t va[] = { 2939407281u,3688426774u,2978577992u,816164u }  ;
              uint32_t vb[] = { 3100715563u,2485608121u,51410437u,5918339u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -13 );
        }
                {
              uint32_t va[] = { 2771982683u,2525242324u,543850589u,4907509u }  ;
              uint32_t vb[] = { 4015740113u,1221843259u,2570753910u,3997538u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -13 );
        }
                {
              uint32_t va[] = { 1876719429u,4069972523u,267990283u,6594247u }  ;
              uint32_t vb[] = { 3718437402u,11027180u,3851462032u,6208624u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -15 );
        }
                {
              uint32_t va[] = { 4244106901u,837869899u,2018339824u,3784619u }  ;
              uint32_t vb[] = { 705847940u,846919519u,2834025127u,6626000u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  11 );
        }
                {
              uint32_t va[] = { 3930215275u,330369101u,2307185861u,1486231u }  ;
              uint32_t vb[] = { 656831953u,2732294685u,3562702646u,260830u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -5 );
        }
                {
              uint32_t va[] = { 742883586u,2679930816u,1713345508u,1481391u }  ;
              uint32_t vb[] = { 1253863621u,3917911817u,1006521064u,4807955u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -21 );
        }
                {
              uint32_t va[] = { 4249038140u,4242289133u,1799384220u,1857060u }  ;
              uint32_t vb[] = { 546958532u,85133848u,606663819u,7925713u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -23 );
        }
                {
              uint32_t va[] = { 2061579039u,1159757314u,3979349852u,2299077u }  ;
              uint32_t vb[] = { 774912340u,2741059100u,1770728599u,3349186u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  23 );
        }
                {
              uint32_t va[] = { 3369288618u,159669581u,2915727419u,350260u }  ;
              uint32_t vb[] = { 2412072995u,1653326945u,3660576633u,105010u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  13 );
        }
                {
              uint32_t va[] = { 3658294519u,228091766u,762096136u,7808996u }  ;
              uint32_t vb[] = { 1198190136u,68992860u,2850662241u,5382476u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  17 );
        }
                {
              uint32_t va[] = { 4169135918u,1703536726u,3019158200u,7467532u }  ;
              uint32_t vb[] = { 2271471094u,1997537704u,29080982u,1370494u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -7 );
        }
                {
              uint32_t va[] = { 3824895988u,1800003245u,567017504u,6644462u }  ;
              uint32_t vb[] = { 3843496577u,1551063479u,2887330175u,4234584u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -15 );
        }
                {
              uint32_t va[] = { 971330300u,2736351739u,60907694u,2161563u }  ;
              uint32_t vb[] = { 4275220412u,3071004996u,3761996593u,6452591u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -9 );
        }
                {
              uint32_t va[] = { 4250184515u,1943761598u,1918437315u,5267579u }  ;
              uint32_t vb[] = { 3990336807u,4059770658u,2190096116u,4604391u }  ;
              size_t l = 119;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  21 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint32_t*, const uint32_t*, size_t), , dim=24") {
                {
              uint32_t va[] = { 12546608u }  ;
              uint32_t vb[] = { 5717486u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 1003863u }  ;
              uint32_t vb[] = { 14526800u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 8992365u }  ;
              uint32_t vb[] = { 6911694u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 5406276u }  ;
              uint32_t vb[] = { 13575657u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 13804200u }  ;
              uint32_t vb[] = { 6894431u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 7892550u }  ;
              uint32_t vb[] = { 2507971u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 4244285u }  ;
              uint32_t vb[] = { 12619554u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 14381925u }  ;
              uint32_t vb[] = { 11292957u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 16505597u }  ;
              uint32_t vb[] = { 7456018u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 5968771u }  ;
              uint32_t vb[] = { 13919825u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 7406489u }  ;
              uint32_t vb[] = { 1737453u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 6343473u }  ;
              uint32_t vb[] = { 6286510u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 13004062u }  ;
              uint32_t vb[] = { 5772517u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 14112523u }  ;
              uint32_t vb[] = { 3337334u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 8623169u }  ;
              uint32_t vb[] = { 11044250u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 15162595u }  ;
              uint32_t vb[] = { 3585994u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 4792081u }  ;
              uint32_t vb[] = { 3856089u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 2134528u }  ;
              uint32_t vb[] = { 5537829u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 3745054u }  ;
              uint32_t vb[] = { 4289800u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 8393511u }  ;
              uint32_t vb[] = { 12145922u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 2997346u }  ;
              uint32_t vb[] = { 7618745u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 13675747u }  ;
              uint32_t vb[] = { 6892785u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 13695150u }  ;
              uint32_t vb[] = { 5562890u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 12650793u }  ;
              uint32_t vb[] = { 2744218u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 4338428u }  ;
              uint32_t vb[] = { 11781421u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 675625u }  ;
              uint32_t vb[] = { 11708987u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 12749631u }  ;
              uint32_t vb[] = { 14507617u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 9019842u }  ;
              uint32_t vb[] = { 12315004u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 9346430u }  ;
              uint32_t vb[] = { 1672137u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 4572582u }  ;
              uint32_t vb[] = { 3700633u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 12199494u }  ;
              uint32_t vb[] = { 10579525u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 7339631u }  ;
              uint32_t vb[] = { 9959376u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 13237195u }  ;
              uint32_t vb[] = { 13935172u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 5430689u }  ;
              uint32_t vb[] = { 843888u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 8750220u }  ;
              uint32_t vb[] = { 5897691u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 16472635u }  ;
              uint32_t vb[] = { 15062508u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 10590065u }  ;
              uint32_t vb[] = { 14418916u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 7057945u }  ;
              uint32_t vb[] = { 1897363u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 7166018u }  ;
              uint32_t vb[] = { 15494001u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 5038076u }  ;
              uint32_t vb[] = { 7404287u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 9475157u }  ;
              uint32_t vb[] = { 773532u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 10164746u }  ;
              uint32_t vb[] = { 8711760u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 760841u }  ;
              uint32_t vb[] = { 14757972u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 2679830u }  ;
              uint32_t vb[] = { 2878743u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  14 );
        }
                {
              uint32_t va[] = { 13318702u }  ;
              uint32_t vb[] = { 2283452u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 14432356u }  ;
              uint32_t vb[] = { 5717104u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  10 );
        }
                {
              uint32_t va[] = { 7745132u }  ;
              uint32_t vb[] = { 10478200u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 8892070u }  ;
              uint32_t vb[] = { 1481920u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 11583791u }  ;
              uint32_t vb[] = { 15940644u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 5742500u }  ;
              uint32_t vb[] = { 9426688u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 7498658u }  ;
              uint32_t vb[] = { 15064852u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 3459378u }  ;
              uint32_t vb[] = { 13238614u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 7018889u }  ;
              uint32_t vb[] = { 1591932u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 11275231u }  ;
              uint32_t vb[] = { 4165590u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 5767015u }  ;
              uint32_t vb[] = { 10193739u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 10945870u }  ;
              uint32_t vb[] = { 13410331u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 11900384u }  ;
              uint32_t vb[] = { 4391482u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 13794912u }  ;
              uint32_t vb[] = { 14570697u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 15549492u }  ;
              uint32_t vb[] = { 16048392u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 1766943u }  ;
              uint32_t vb[] = { 7617770u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 3393826u }  ;
              uint32_t vb[] = { 14273258u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 12447756u }  ;
              uint32_t vb[] = { 12070503u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 7023523u }  ;
              uint32_t vb[] = { 4320794u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 7939057u }  ;
              uint32_t vb[] = { 4037011u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 12405205u }  ;
              uint32_t vb[] = { 1287805u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 4546445u }  ;
              uint32_t vb[] = { 15010289u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 16543834u }  ;
              uint32_t vb[] = { 11330813u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 15348994u }  ;
              uint32_t vb[] = { 490896u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 10462323u }  ;
              uint32_t vb[] = { 14992518u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 7173534u }  ;
              uint32_t vb[] = { 9507937u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 16536935u }  ;
              uint32_t vb[] = { 1699274u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 5042690u }  ;
              uint32_t vb[] = { 6968881u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 12959175u }  ;
              uint32_t vb[] = { 15942929u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 9844637u }  ;
              uint32_t vb[] = { 2691045u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 13733268u }  ;
              uint32_t vb[] = { 9402071u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 14090194u }  ;
              uint32_t vb[] = { 3840523u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 7682508u }  ;
              uint32_t vb[] = { 2560247u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 4179792u }  ;
              uint32_t vb[] = { 16431713u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 2266485u }  ;
              uint32_t vb[] = { 2417552u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 11363282u }  ;
              uint32_t vb[] = { 10637152u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 8181441u }  ;
              uint32_t vb[] = { 11639921u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 14781033u }  ;
              uint32_t vb[] = { 11663941u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 2817812u }  ;
              uint32_t vb[] = { 14406991u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 3951957u }  ;
              uint32_t vb[] = { 2348238u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 15575054u }  ;
              uint32_t vb[] = { 11968914u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 12476871u }  ;
              uint32_t vb[] = { 7459305u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 1632106u }  ;
              uint32_t vb[] = { 14005997u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 13847114u }  ;
              uint32_t vb[] = { 13528672u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 6995440u }  ;
              uint32_t vb[] = { 5353986u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 12408184u }  ;
              uint32_t vb[] = { 8006653u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 6637234u }  ;
              uint32_t vb[] = { 8845265u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 860614u }  ;
              uint32_t vb[] = { 10270384u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 15196254u }  ;
              uint32_t vb[] = { 7653967u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 13326134u }  ;
              uint32_t vb[] = { 3375642u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 2639186u }  ;
              uint32_t vb[] = { 11133673u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 4599815u }  ;
              uint32_t vb[] = { 2105146u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 8351634u }  ;
              uint32_t vb[] = { 412650u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 5575158u }  ;
              uint32_t vb[] = { 13775108u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 16179685u }  ;
              uint32_t vb[] = { 4415214u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 3950526u }  ;
              uint32_t vb[] = { 4427466u }  ;
              size_t l = 24;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint32_t*, const uint32_t*, size_t), , dim=23") {
                {
              uint32_t va[] = { 6433926u }  ;
              uint32_t vb[] = { 503453u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  3 );
        }
                {
              uint32_t va[] = { 2112914u }  ;
              uint32_t vb[] = { 3379661u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -3 );
        }
                {
              uint32_t va[] = { 7985984u }  ;
              uint32_t vb[] = { 1014370u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  3 );
        }
                {
              uint32_t va[] = { 1129456u }  ;
              uint32_t vb[] = { 2276127u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -1 );
        }
                {
              uint32_t va[] = { 4034002u }  ;
              uint32_t vb[] = { 6549527u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -7 );
        }
                {
              uint32_t va[] = { 3796575u }  ;
              uint32_t vb[] = { 5892110u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  9 );
        }
                {
              uint32_t va[] = { 7787564u }  ;
              uint32_t vb[] = { 6121270u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -1 );
        }
                {
              uint32_t va[] = { 1857281u }  ;
              uint32_t vb[] = { 4749413u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  1 );
        }
                {
              uint32_t va[] = { 3289689u }  ;
              uint32_t vb[] = { 727988u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -3 );
        }
                {
              uint32_t va[] = { 4206730u }  ;
              uint32_t vb[] = { 4058702u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -7 );
        }
                {
              uint32_t va[] = { 4785623u }  ;
              uint32_t vb[] = { 5039573u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  11 );
        }
                {
              uint32_t va[] = { 635115u }  ;
              uint32_t vb[] = { 6138875u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  7 );
        }
                {
              uint32_t va[] = { 1032246u }  ;
              uint32_t vb[] = { 2590916u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  5 );
        }
                {
              uint32_t va[] = { 1060101u }  ;
              uint32_t vb[] = { 2870064u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -5 );
        }
                {
              uint32_t va[] = { 7191820u }  ;
              uint32_t vb[] = { 3900874u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  1 );
        }
                {
              uint32_t va[] = { 1616588u }  ;
              uint32_t vb[] = { 7950804u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -1 );
        }
                {
              uint32_t va[] = { 5707429u }  ;
              uint32_t vb[] = { 4524064u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  5 );
        }
                {
              uint32_t va[] = { 2148247u }  ;
              uint32_t vb[] = { 3357258u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -9 );
        }
                {
              uint32_t va[] = { 3409189u }  ;
              uint32_t vb[] = { 3903146u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -3 );
        }
                {
              uint32_t va[] = { 237236u }  ;
              uint32_t vb[] = { 8063551u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  1 );
        }
                {
              uint32_t va[] = { 2861814u }  ;
              uint32_t vb[] = { 2894961u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  3 );
        }
                {
              uint32_t va[] = { 3809836u }  ;
              uint32_t vb[] = { 6711316u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  3 );
        }
                {
              uint32_t va[] = { 6650313u }  ;
              uint32_t vb[] = { 387493u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  7 );
        }
                {
              uint32_t va[] = { 1523369u }  ;
              uint32_t vb[] = { 6081806u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -7 );
        }
                {
              uint32_t va[] = { 7691883u }  ;
              uint32_t vb[] = { 4786418u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -1 );
        }
                {
              uint32_t va[] = { 1343100u }  ;
              uint32_t vb[] = { 7583246u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -1 );
        }
                {
              uint32_t va[] = { 1390322u }  ;
              uint32_t vb[] = { 2577934u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  1 );
        }
                {
              uint32_t va[] = { 5144164u }  ;
              uint32_t vb[] = { 1533190u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  1 );
        }
                {
              uint32_t va[] = { 5802709u }  ;
              uint32_t vb[] = { 4272009u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  1 );
        }
                {
              uint32_t va[] = { 7047309u }  ;
              uint32_t vb[] = { 3789084u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  3 );
        }
                {
              uint32_t va[] = { 5372093u }  ;
              uint32_t vb[] = { 8267309u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -3 );
        }
                {
              uint32_t va[] = { 620332u }  ;
              uint32_t vb[] = { 7606730u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -7 );
        }
                {
              uint32_t va[] = { 1744146u }  ;
              uint32_t vb[] = { 5793176u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  3 );
        }
                {
              uint32_t va[] = { 7078243u }  ;
              uint32_t vb[] = { 2771131u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  3 );
        }
                {
              uint32_t va[] = { 4279409u }  ;
              uint32_t vb[] = { 4614702u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -1 );
        }
                {
              uint32_t va[] = { 5286069u }  ;
              uint32_t vb[] = { 7205286u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -1 );
        }
                {
              uint32_t va[] = { 1451957u }  ;
              uint32_t vb[] = { 7887480u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -9 );
        }
                {
              uint32_t va[] = { 3210867u }  ;
              uint32_t vb[] = { 184043u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  7 );
        }
                {
              uint32_t va[] = { 5204286u }  ;
              uint32_t vb[] = { 4102502u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  1 );
        }
                {
              uint32_t va[] = { 5712042u }  ;
              uint32_t vb[] = { 4734145u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -1 );
        }
                {
              uint32_t va[] = { 8150140u }  ;
              uint32_t vb[] = { 1431434u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -5 );
        }
                {
              uint32_t va[] = { 1488358u }  ;
              uint32_t vb[] = { 675085u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -5 );
        }
                {
              uint32_t va[] = { 3589288u }  ;
              uint32_t vb[] = { 3785481u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  3 );
        }
                {
              uint32_t va[] = { 5652987u }  ;
              uint32_t vb[] = { 1184673u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  5 );
        }
                {
              uint32_t va[] = { 1172371u }  ;
              uint32_t vb[] = { 2665832u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -7 );
        }
                {
              uint32_t va[] = { 5307518u }  ;
              uint32_t vb[] = { 2179600u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -7 );
        }
                {
              uint32_t va[] = { 1619983u }  ;
              uint32_t vb[] = { 5221344u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -7 );
        }
                {
              uint32_t va[] = { 4936918u }  ;
              uint32_t vb[] = { 2467644u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -9 );
        }
                {
              uint32_t va[] = { 4039434u }  ;
              uint32_t vb[] = { 3416888u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  11 );
        }
                {
              uint32_t va[] = { 3042307u }  ;
              uint32_t vb[] = { 1318077u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -3 );
        }
                {
              uint32_t va[] = { 5331069u }  ;
              uint32_t vb[] = { 3756292u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  3 );
        }
                {
              uint32_t va[] = { 7946895u }  ;
              uint32_t vb[] = { 7671322u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  5 );
        }
                {
              uint32_t va[] = { 3777401u }  ;
              uint32_t vb[] = { 7271340u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -1 );
        }
                {
              uint32_t va[] = { 3679674u }  ;
              uint32_t vb[] = { 6933869u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -3 );
        }
                {
              uint32_t va[] = { 3424622u }  ;
              uint32_t vb[] = { 485387u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  1 );
        }
                {
              uint32_t va[] = { 116817u }  ;
              uint32_t vb[] = { 5390794u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -1 );
        }
                {
              uint32_t va[] = { 8356826u }  ;
              uint32_t vb[] = { 4585883u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  1 );
        }
                {
              uint32_t va[] = { 1233330u }  ;
              uint32_t vb[] = { 7236802u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -5 );
        }
                {
              uint32_t va[] = { 7002057u }  ;
              uint32_t vb[] = { 7842827u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -3 );
        }
                {
              uint32_t va[] = { 5757833u }  ;
              uint32_t vb[] = { 5948298u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  9 );
        }
                {
              uint32_t va[] = { 4586602u }  ;
              uint32_t vb[] = { 8245841u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -1 );
        }
                {
              uint32_t va[] = { 3469334u }  ;
              uint32_t vb[] = { 3838082u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  7 );
        }
                {
              uint32_t va[] = { 4437826u }  ;
              uint32_t vb[] = { 7960528u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  5 );
        }
                {
              uint32_t va[] = { 6326615u }  ;
              uint32_t vb[] = { 1775537u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -9 );
        }
                {
              uint32_t va[] = { 1453247u }  ;
              uint32_t vb[] = { 2773691u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  1 );
        }
                {
              uint32_t va[] = { 2757878u }  ;
              uint32_t vb[] = { 621235u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  1 );
        }
                {
              uint32_t va[] = { 6692933u }  ;
              uint32_t vb[] = { 2890875u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -1 );
        }
                {
              uint32_t va[] = { 464959u }  ;
              uint32_t vb[] = { 1251973u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  7 );
        }
                {
              uint32_t va[] = { 1729864u }  ;
              uint32_t vb[] = { 5113603u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  3 );
        }
                {
              uint32_t va[] = { 955372u }  ;
              uint32_t vb[] = { 903129u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  5 );
        }
                {
              uint32_t va[] = { 5518954u }  ;
              uint32_t vb[] = { 8386127u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  5 );
        }
                {
              uint32_t va[] = { 5859661u }  ;
              uint32_t vb[] = { 8363247u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -1 );
        }
                {
              uint32_t va[] = { 1821417u }  ;
              uint32_t vb[] = { 3085794u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -3 );
        }
                {
              uint32_t va[] = { 298375u }  ;
              uint32_t vb[] = { 3555781u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  5 );
        }
                {
              uint32_t va[] = { 7685761u }  ;
              uint32_t vb[] = { 5927252u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -7 );
        }
                {
              uint32_t va[] = { 2917286u }  ;
              uint32_t vb[] = { 178542u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  1 );
        }
                {
              uint32_t va[] = { 1225226u }  ;
              uint32_t vb[] = { 6734205u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -7 );
        }
                {
              uint32_t va[] = { 5732211u }  ;
              uint32_t vb[] = { 5645607u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  9 );
        }
                {
              uint32_t va[] = { 5570255u }  ;
              uint32_t vb[] = { 3818022u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -7 );
        }
                {
              uint32_t va[] = { 574453u }  ;
              uint32_t vb[] = { 5946907u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -7 );
        }
                {
              uint32_t va[] = { 502389u }  ;
              uint32_t vb[] = { 6121642u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -5 );
        }
                {
              uint32_t va[] = { 1326877u }  ;
              uint32_t vb[] = { 4196274u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -1 );
        }
                {
              uint32_t va[] = { 8242870u }  ;
              uint32_t vb[] = { 3007526u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  11 );
        }
                {
              uint32_t va[] = { 1483415u }  ;
              uint32_t vb[] = { 2372169u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  1 );
        }
                {
              uint32_t va[] = { 6430820u }  ;
              uint32_t vb[] = { 3498431u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -3 );
        }
                {
              uint32_t va[] = { 3157208u }  ;
              uint32_t vb[] = { 53732u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -3 );
        }
                {
              uint32_t va[] = { 2538010u }  ;
              uint32_t vb[] = { 2633580u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -1 );
        }
                {
              uint32_t va[] = { 475981u }  ;
              uint32_t vb[] = { 3742763u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -7 );
        }
                {
              uint32_t va[] = { 4131755u }  ;
              uint32_t vb[] = { 658725u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  3 );
        }
                {
              uint32_t va[] = { 3244123u }  ;
              uint32_t vb[] = { 2023395u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -7 );
        }
                {
              uint32_t va[] = { 2707985u }  ;
              uint32_t vb[] = { 981988u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -5 );
        }
                {
              uint32_t va[] = { 4916227u }  ;
              uint32_t vb[] = { 2797845u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  1 );
        }
                {
              uint32_t va[] = { 5596196u }  ;
              uint32_t vb[] = { 5247626u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -1 );
        }
                {
              uint32_t va[] = { 1702821u }  ;
              uint32_t vb[] = { 5044733u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  7 );
        }
                {
              uint32_t va[] = { 7902387u }  ;
              uint32_t vb[] = { 2085623u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  3 );
        }
                {
              uint32_t va[] = { 5762842u }  ;
              uint32_t vb[] = { 6789369u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -3 );
        }
                {
              uint32_t va[] = { 5630307u }  ;
              uint32_t vb[] = { 5391080u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -1 );
        }
                {
              uint32_t va[] = { 7605162u }  ;
              uint32_t vb[] = { 2607197u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -7 );
        }
                {
              uint32_t va[] = { 3148372u }  ;
              uint32_t vb[] = { 4304127u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -3 );
        }
                {
              uint32_t va[] = { 5849160u }  ;
              uint32_t vb[] = { 8150256u }  ;
              size_t l = 23;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  3 );
        }
            }

        SECTION("int dot_prod_cb_cb_enc(const uint32_t*, const uint32_t*, size_t), , dim=150") {
                {
              uint32_t va[] = { 3754985942u,516257353u,1942211551u,632398536u,3659474u }  ;
              uint32_t vb[] = { 1028543205u,360462753u,842012038u,1810360076u,95153u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 3030651890u,1686329765u,4123231624u,237575116u,3423168u }  ;
              uint32_t vb[] = { 120971642u,3237105747u,2841276015u,2180038226u,3459716u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 871381136u,794617979u,3937849059u,3256876111u,3578497u }  ;
              uint32_t vb[] = { 858598426u,1814368538u,3381592412u,525795987u,4160352u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 1874090213u,829518003u,2052804258u,2126769508u,3769929u }  ;
              uint32_t vb[] = { 1206343421u,2385666005u,4178364581u,996257591u,2625516u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  20 );
        }
                {
              uint32_t va[] = { 2915085659u,1730889607u,433683555u,1320493909u,862255u }  ;
              uint32_t vb[] = { 165424302u,1278276176u,3566485091u,3931629170u,1020654u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  14 );
        }
                {
              uint32_t va[] = { 4101191091u,3237457852u,253163095u,2026708458u,22662u }  ;
              uint32_t vb[] = { 3950181020u,3252092231u,3644130255u,4208173632u,2325108u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 1745517764u,1006604041u,1859706571u,48921110u,3576876u }  ;
              uint32_t vb[] = { 1652962271u,1052739700u,279476851u,1323042440u,1882838u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 925180454u,2956163502u,1217313286u,2952555204u,3154390u }  ;
              uint32_t vb[] = { 516640166u,1770702135u,2060433576u,3991969309u,1379952u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 2156127145u,3819168520u,3081256874u,2666511366u,3194257u }  ;
              uint32_t vb[] = { 394641546u,3911747624u,3471015356u,3246131597u,1975252u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  10 );
        }
                {
              uint32_t va[] = { 3796508079u,4245307263u,2781776573u,1121242220u,3549918u }  ;
              uint32_t vb[] = { 1513792901u,1229423686u,102759208u,2604342576u,2111042u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 971021348u,3339707894u,1796735426u,181468042u,2049153u }  ;
              uint32_t vb[] = { 4038168405u,2657109997u,2652992126u,2078119944u,1561253u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 1012602855u,2245279497u,1478955070u,3436361960u,3014327u }  ;
              uint32_t vb[] = { 768973872u,2703282358u,1227027230u,1669577514u,2617845u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  12 );
        }
                {
              uint32_t va[] = { 4268591399u,596659385u,1711806227u,3245300542u,1237628u }  ;
              uint32_t vb[] = { 2441333984u,1693255859u,2837046077u,3081589785u,2650527u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 3033451981u,1440573748u,1301674733u,1049569224u,1787726u }  ;
              uint32_t vb[] = { 3652641297u,1778046950u,3769714889u,861325302u,3265492u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 815392561u,2464343889u,2523694203u,3060795758u,2556716u }  ;
              uint32_t vb[] = { 1259728981u,1004756920u,1181722602u,416235219u,3983064u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 3803509131u,2209265255u,433889986u,701991470u,1548517u }  ;
              uint32_t vb[] = { 4099050166u,4128430139u,3087234048u,2894216381u,1484598u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -12 );
        }
                {
              uint32_t va[] = { 665460904u,67178583u,1670459135u,1315532641u,1147485u }  ;
              uint32_t vb[] = { 2401984098u,2610149238u,3724744984u,1505498579u,1693152u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -12 );
        }
                {
              uint32_t va[] = { 1604936630u,3813011315u,2772796012u,1340967145u,1566669u }  ;
              uint32_t vb[] = { 3182060744u,187978217u,1192756543u,1820426227u,644083u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 598627512u,2829741715u,2680342871u,1877315538u,3507151u }  ;
              uint32_t vb[] = { 3067634525u,1068127194u,1194113869u,3760440400u,819283u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 1722974178u,1583314178u,4282911052u,52901716u,2302311u }  ;
              uint32_t vb[] = { 1900078262u,1346346537u,3928734240u,90300203u,211717u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 517095737u,4086990493u,1479208385u,3649295665u,3359958u }  ;
              uint32_t vb[] = { 2016391473u,734813132u,1883916414u,1972717586u,2246960u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 2052502379u,1466973366u,1408346948u,2604339948u,1556396u }  ;
              uint32_t vb[] = { 2119068380u,3364736792u,751519373u,1401513190u,51067u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -24 );
        }
                {
              uint32_t va[] = { 1528149619u,3220178669u,4189388464u,1920724789u,1071817u }  ;
              uint32_t vb[] = { 2045855366u,3552617335u,1436179192u,3212306169u,1785218u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  14 );
        }
                {
              uint32_t va[] = { 1742256776u,3985323398u,108196359u,456368256u,963874u }  ;
              uint32_t vb[] = { 1265720161u,76844770u,790304960u,2587082732u,1781103u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 1040923152u,1706640422u,452288024u,4222295135u,3766741u }  ;
              uint32_t vb[] = { 655631577u,498807104u,1551038437u,408301757u,4030158u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 3190016078u,1283406496u,1875479343u,2363610280u,1071738u }  ;
              uint32_t vb[] = { 1124366184u,1001452866u,2101997464u,3125553228u,1923902u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 863641896u,1269532469u,2966811530u,3160544842u,2764140u }  ;
              uint32_t vb[] = { 3686318212u,3825642215u,469790491u,1805618237u,3292465u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -12 );
        }
                {
              uint32_t va[] = { 322550979u,2562903217u,3569833533u,3340323707u,20788u }  ;
              uint32_t vb[] = { 1989301050u,3824377628u,257546415u,1962197037u,86117u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -14 );
        }
                {
              uint32_t va[] = { 4109868602u,3391610347u,2658465317u,92541658u,3342467u }  ;
              uint32_t vb[] = { 3482531015u,35295396u,2169695799u,1128440546u,144904u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 3642108697u,1556611661u,3585638652u,718520118u,3362872u }  ;
              uint32_t vb[] = { 2589758496u,3128387300u,1084922365u,24651545u,823586u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 1186919313u,3531716092u,2108326467u,1562927674u,2319838u }  ;
              uint32_t vb[] = { 875773261u,1640512545u,3503135347u,3473537139u,3980773u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 1478893647u,2980005757u,4009661342u,2869158088u,3968759u }  ;
              uint32_t vb[] = { 1383292715u,1483122371u,1579555603u,3156629340u,3124343u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 4123583932u,2084805604u,3449923617u,209308000u,1147357u }  ;
              uint32_t vb[] = { 884637001u,2310806836u,3004691557u,2791774444u,4173871u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 1271323268u,546617937u,650013502u,1651247789u,3213458u }  ;
              uint32_t vb[] = { 2188315771u,3017024891u,1247069370u,983453414u,3651136u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 2658386836u,1042906019u,2921169625u,13518565u,815828u }  ;
              uint32_t vb[] = { 3722616939u,3140891360u,3302118871u,3725315597u,2701865u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 180494760u,2783722335u,2078909695u,1011874268u,2376417u }  ;
              uint32_t vb[] = { 3281673708u,7819484u,1039007717u,4187015812u,1659315u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 707850235u,1173993626u,2374267552u,3422383355u,278460u }  ;
              uint32_t vb[] = { 1063799763u,1173821370u,2129304609u,1313841451u,2953085u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  24 );
        }
                {
              uint32_t va[] = { 3553254818u,2128751374u,813616790u,355222142u,3034262u }  ;
              uint32_t vb[] = { 1633808946u,3699532256u,3319123332u,1084945495u,1745089u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 4094715619u,750844562u,1378340452u,4082210109u,1502468u }  ;
              uint32_t vb[] = { 140444190u,2674355824u,3568778175u,2207266595u,2041182u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 1612171015u,3462569586u,1228865568u,3996836543u,2110334u }  ;
              uint32_t vb[] = { 193008096u,2275978464u,1181898063u,71778230u,3422648u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 4263226234u,214699615u,3293005155u,52555932u,2604650u }  ;
              uint32_t vb[] = { 4125545007u,2150580773u,2029332690u,2023218939u,3691478u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -28 );
        }
                {
              uint32_t va[] = { 3447141490u,4168301960u,3986391269u,3147857821u,3507567u }  ;
              uint32_t vb[] = { 1892567113u,1944985634u,4175597829u,3396234536u,1625498u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 1535920493u,1253451540u,3191201439u,2504521151u,1076271u }  ;
              uint32_t vb[] = { 3945231050u,3763184096u,2993689388u,3924866257u,2184226u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -8 );
        }
                {
              uint32_t va[] = { 2291283511u,1614050994u,1770579546u,1153090744u,1079569u }  ;
              uint32_t vb[] = { 24342691u,1282242686u,2019751064u,3082554765u,3790007u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  16 );
        }
                {
              uint32_t va[] = { 235504802u,4109851788u,1286509416u,2718797404u,3271827u }  ;
              uint32_t vb[] = { 2764904079u,14375654u,1574933160u,946606108u,3792738u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  30 );
        }
                {
              uint32_t va[] = { 3229740176u,3563288705u,3195792242u,1573949568u,3386610u }  ;
              uint32_t vb[] = { 1331187064u,4224070842u,3963863142u,635078745u,3642198u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  12 );
        }
                {
              uint32_t va[] = { 1576793176u,374321291u,1806227326u,3426359773u,1094338u }  ;
              uint32_t vb[] = { 1505814148u,2122800199u,3666245618u,26905812u,2101564u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  16 );
        }
                {
              uint32_t va[] = { 2529310245u,535700367u,69801185u,3648606276u,4091368u }  ;
              uint32_t vb[] = { 3581545453u,161617855u,538684315u,3007208550u,2011054u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  26 );
        }
                {
              uint32_t va[] = { 2301835202u,1615608340u,2895840297u,2304708272u,1432336u }  ;
              uint32_t vb[] = { 1963200658u,3721597599u,2127576671u,1249802497u,2098208u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -12 );
        }
                {
              uint32_t va[] = { 1776212701u,538219416u,3530003738u,1374619375u,919531u }  ;
              uint32_t vb[] = { 1981341001u,266436103u,2079985844u,3278901881u,3170108u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -18 );
        }
                {
              uint32_t va[] = { 3626092392u,2494201555u,2369753924u,1104040224u,52250u }  ;
              uint32_t vb[] = { 2100610311u,3009592604u,4042650176u,1830304011u,3208181u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 604743980u,2289821966u,3793683864u,1477957644u,2044227u }  ;
              uint32_t vb[] = { 1329357566u,1708947212u,2958171172u,2306169805u,2631872u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 527813632u,3242432328u,2023417145u,3766090888u,2494539u }  ;
              uint32_t vb[] = { 1006198687u,2029466604u,325051197u,468973584u,3102801u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  24 );
        }
                {
              uint32_t va[] = { 2020896899u,4001964228u,1683635322u,3010387486u,2499482u }  ;
              uint32_t vb[] = { 1671995197u,2866751745u,247048541u,4048471137u,3076471u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 1665986634u,1918757436u,2919925825u,4257083693u,2585235u }  ;
              uint32_t vb[] = { 1409816388u,2373634546u,830321907u,2586301333u,3503084u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -12 );
        }
                {
              uint32_t va[] = { 3548264209u,1214539082u,2844411703u,1937798529u,2582469u }  ;
              uint32_t vb[] = { 3848179605u,2660941635u,1744590048u,2706099187u,3662526u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 2634458846u,4076417052u,4168205286u,19400105u,923470u }  ;
              uint32_t vb[] = { 3072753190u,2437359428u,2509205036u,3703765706u,4047516u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -24 );
        }
                {
              uint32_t va[] = { 3962209855u,1296323716u,2307396576u,2518635257u,2350627u }  ;
              uint32_t vb[] = { 2656461650u,870449901u,3692787210u,2761016186u,2416456u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 999641627u,2527922919u,3022569512u,1674490578u,3772774u }  ;
              uint32_t vb[] = { 213283219u,1485099249u,1577645808u,3655456557u,3975135u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 527361032u,2996328202u,1896740071u,2860046199u,2140329u }  ;
              uint32_t vb[] = { 609324517u,3821563038u,672375217u,1766963940u,3708145u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 2810883287u,4245801155u,3570134989u,3726283068u,703264u }  ;
              uint32_t vb[] = { 1828667880u,1247519287u,976289080u,2099219354u,2791404u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -14 );
        }
                {
              uint32_t va[] = { 258328731u,1901614257u,1191563137u,3384139520u,637349u }  ;
              uint32_t vb[] = { 2988599888u,3582911094u,3114665065u,3553427652u,4126164u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -20 );
        }
                {
              uint32_t va[] = { 772223162u,1367462541u,16581917u,3169829991u,984268u }  ;
              uint32_t vb[] = { 1613547504u,2570686967u,4187881117u,3507265133u,2282383u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 2171937345u,3316358491u,3826578615u,2393896438u,2371061u }  ;
              uint32_t vb[] = { 149967430u,1679983586u,3638531558u,3278131623u,3721894u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  10 );
        }
                {
              uint32_t va[] = { 250805291u,2283832998u,3480269985u,2680310716u,2126392u }  ;
              uint32_t vb[] = { 2748503904u,38421898u,1069496960u,2053262664u,374452u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  16 );
        }
                {
              uint32_t va[] = { 3797451794u,2896994904u,3212993618u,3159823554u,3865635u }  ;
              uint32_t vb[] = { 1594261054u,2771401003u,1842973683u,617593585u,3546348u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -12 );
        }
                {
              uint32_t va[] = { 713335588u,118908384u,3868189575u,2386550549u,3482763u }  ;
              uint32_t vb[] = { 546085499u,2861586965u,2998269793u,3764078341u,4108177u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  12 );
        }
                {
              uint32_t va[] = { 1822209300u,3859346619u,1681827124u,776665508u,3704949u }  ;
              uint32_t vb[] = { 3939890056u,3495611692u,2326890040u,3282589037u,4186954u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 3316013843u,396599715u,253780538u,2045073487u,3872741u }  ;
              uint32_t vb[] = { 2497856836u,1062241498u,497350101u,1521359768u,44965u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 3948622586u,3656054328u,2097195521u,3339581681u,105085u }  ;
              uint32_t vb[] = { 2714460801u,2676783286u,234742584u,4090307810u,3959243u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 1005648405u,750583631u,497601354u,2556230291u,3580194u }  ;
              uint32_t vb[] = { 3921965539u,3956963413u,543503429u,2445258867u,1219220u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 836951969u,2827310333u,3006671931u,1515098201u,1124832u }  ;
              uint32_t vb[] = { 4226235846u,1636536667u,3484759746u,121357963u,1594392u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 3916988805u,2503108611u,3769232966u,2209735940u,2303795u }  ;
              uint32_t vb[] = { 3913127432u,999532168u,3125096059u,234418391u,2874750u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 3305213195u,1490682683u,3268736018u,4206727780u,1693634u }  ;
              uint32_t vb[] = { 2891053111u,448526211u,3761098057u,3727948485u,2669093u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  6 );
        }
                {
              uint32_t va[] = { 801406963u,1163750817u,1188559642u,3512291244u,3701875u }  ;
              uint32_t vb[] = { 2179850870u,4071584408u,3030795617u,3569815534u,2426286u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -14 );
        }
                {
              uint32_t va[] = { 3149820765u,897598453u,990289443u,2299379957u,919698u }  ;
              uint32_t vb[] = { 4152435969u,1287262845u,3397462869u,3155225573u,1172074u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 3900541721u,3468664167u,3317394117u,3756058805u,3137956u }  ;
              uint32_t vb[] = { 2553219362u,3628573517u,3728089851u,3519145787u,505546u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  12 );
        }
                {
              uint32_t va[] = { 2334352352u,1149562291u,114407694u,3605485218u,1716559u }  ;
              uint32_t vb[] = { 1503174921u,2706235373u,3630415549u,2429175052u,1017617u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -12 );
        }
                {
              uint32_t va[] = { 680358983u,2410494411u,3181496275u,1483118461u,873107u }  ;
              uint32_t vb[] = { 2044459148u,2988205711u,3131020445u,3108595680u,3013487u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 2761012598u,1767030239u,3077662516u,4001995718u,2620224u }  ;
              uint32_t vb[] = { 488017345u,4239455354u,1799642632u,2752647917u,1937230u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -2 );
        }
                {
              uint32_t va[] = { 1023049657u,2296573743u,1766006021u,1164681399u,2739464u }  ;
              uint32_t vb[] = { 2230179608u,2202198586u,1164084630u,1958996765u,1927742u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  24 );
        }
                {
              uint32_t va[] = { 107162954u,2299801649u,2503810493u,3660234426u,2327007u }  ;
              uint32_t vb[] = { 2873842294u,1484465329u,577906021u,784863332u,3727462u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -10 );
        }
                {
              uint32_t va[] = { 4060087854u,2464218032u,465413947u,87920839u,562354u }  ;
              uint32_t vb[] = { 1776063794u,3476249330u,3891996402u,2914337075u,993679u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  0 );
        }
                {
              uint32_t va[] = { 3718148555u,2269779519u,1000070050u,1770385058u,4005012u }  ;
              uint32_t vb[] = { 3393131664u,1220761554u,1795247946u,2512260685u,2099989u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 2696721622u,4108179632u,437873700u,1562870111u,3996561u }  ;
              uint32_t vb[] = { 2696936704u,4207387351u,578331695u,2021885724u,1705356u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  24 );
        }
                {
              uint32_t va[] = { 2301527959u,668229341u,2447130372u,2467285456u,1207672u }  ;
              uint32_t vb[] = { 2925105587u,1783254625u,2646243203u,250086725u,256629u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  10 );
        }
                {
              uint32_t va[] = { 4173942370u,1514674951u,498781056u,1552053822u,2276733u }  ;
              uint32_t vb[] = { 3354807322u,1670191873u,3430139356u,1655678852u,2938131u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  8 );
        }
                {
              uint32_t va[] = { 354665927u,4033401382u,1180348326u,393610904u,4020702u }  ;
              uint32_t vb[] = { 1428272224u,3065221067u,3519243225u,3935334321u,1832723u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -16 );
        }
                {
              uint32_t va[] = { 4241623143u,109352167u,3672851621u,2919135501u,1859713u }  ;
              uint32_t vb[] = { 1922688286u,2129006438u,3349195505u,2162790690u,4118097u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 3692305308u,4115876185u,4246007108u,1102045045u,2421100u }  ;
              uint32_t vb[] = { 2474701130u,190787397u,1471576527u,94452484u,1209303u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  4 );
        }
                {
              uint32_t va[] = { 4278452499u,3938547523u,445134451u,93183790u,3834470u }  ;
              uint32_t vb[] = { 2690126459u,1968339362u,3071377388u,1879149780u,861624u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -20 );
        }
                {
              uint32_t va[] = { 244376588u,584535230u,3015486668u,1941841209u,2284469u }  ;
              uint32_t vb[] = { 3297683480u,14576381u,1630164457u,2720490068u,2416132u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  20 );
        }
                {
              uint32_t va[] = { 2562404953u,3064390896u,123693487u,2420702463u,369426u }  ;
              uint32_t vb[] = { 2398470505u,19307705u,3194471563u,327863290u,1877126u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  18 );
        }
                {
              uint32_t va[] = { 3680021690u,3069096828u,1377006223u,1126574062u,2953631u }  ;
              uint32_t vb[] = { 1599817493u,147000986u,2646130212u,410515490u,841212u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -20 );
        }
                {
              uint32_t va[] = { 1087597196u,1343270401u,2393079838u,1339975452u,2027024u }  ;
              uint32_t vb[] = { 688296097u,3659737769u,731688992u,2452513242u,460776u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 1803304758u,1441670275u,4124044122u,1590767542u,2430729u }  ;
              uint32_t vb[] = { 2315302067u,1310322731u,2252547481u,1491211190u,3762832u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  2 );
        }
                {
              uint32_t va[] = { 3444883549u,26911241u,3184589714u,640414162u,631716u }  ;
              uint32_t vb[] = { 371231295u,1354250001u,2334235646u,3127837907u,1669622u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  22 );
        }
                {
              uint32_t va[] = { 856515771u,2954270107u,2193699594u,675917178u,103549u }  ;
              uint32_t vb[] = { 2235697548u,1275219104u,1791132672u,437928797u,2364584u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -6 );
        }
                {
              uint32_t va[] = { 50810273u,3750079265u,689130847u,3925072722u,4006133u }  ;
              uint32_t vb[] = { 3645264454u,2669695491u,3335036069u,1511193353u,3681394u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -4 );
        }
                {
              uint32_t va[] = { 1930044496u,2049435853u,3001703076u,1382283159u,1267529u }  ;
              uint32_t vb[] = { 1793480670u,1207883828u,1866774746u,1066351811u,2775743u }  ;
              size_t l = 150;
              REQUIRE(dot_prod_cb_cb_enc<int /* 32 */, uint32_t>(va, vb, l) ==  -24 );
        }
            }

    }