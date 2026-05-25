import os
import torch
import open_clip
from PIL import Image
from torchvision import transforms

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# =========================
# 1. BioCLIP 权重路径
# =========================
# 你需要把 open_clip_pytorch_model.bin 放到项目里，比如：
# PROJECT/models/bioclip/open_clip_pytorch_model.bin

BIOCLIP_WEIGHT = "model/open_clip_pytorch_model.bin"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 2. 植物种类列表
# 这里先放简化版
# 你可以把同学代码里的 400 个完整 PLANT_SPECIES 复制替换这里
# =========================
PLANT_SPECIES = [
    # ========== 1. 龟背竹与蔓绿绒属 (1-15) ==========
    "Monstera deliciosa (龟背竹)",
    "Monstera adansonii (窗孔龟背竹)",
    "Monstera obliqua (斜叶龟背竹)",
    "Monstera standleyana (星点龟背竹)",
    "Philodendron hederaceum (心叶蔓绿绒)",
    "Philodendron birkin (白锦蔓绿绒)",
    "Philodendron gloriosum (锦缎蔓绿绒)",
    "Philodendron melanochrysum (黑金蔓绿绒)",
    "Philodendron xanadu (仙羽蔓绿绒)",
    "Philodendron selloum (春羽)",
    "Philodendron micans (金丝蔓绿绒)",
    "Philodendron erubescens (红苞蔓绿绒)",
    "Philodendron verrucosum (疣柄蔓绿绒)",
    "Philodendron squamiferum (鳞柄蔓绿绒)",
    "Rhaphidophora tetrasperma (姬龟背)",

    # ========== 2. 绿萝与星点藤 (16-19) ==========
    "Epipremnum aureum (绿萝)",
    "Epipremnum pinnatum (大叶绿萝)",
    "Scindapsus pictus (星点藤)",
    "Scindapsus treubii (暗叶星点藤)",

    # ========== 3. 花烛与白掌 (20-25) ==========
    "Anthurium andraeanum (红掌)",
    "Anthurium clarinervium (明脉花烛)",
    "Anthurium crystallinum (水晶花烛)",
    "Anthurium veitchii (长叶花烛)",
    "Anthurium warocqueanum (女王花烛)",
    "Spathiphyllum wallisii (白掌)",

    # ========== 4. 海芋属 (26-32) ==========
    "Alocasia amazonica (非洲面具海芋)",
    "Alocasia zebrina (斑马海芋)",
    "Alocasia macrorrhiza (大野芋)",
    "Alocasia micholitziana (箭叶海芋)",
    "Alocasia cuprea (铜叶海芋)",
    "Alocasia baginda (龙鳞海芋)",
    "Colocasia esculenta (芋头)",

    # ========== 5. 彩叶芋/合果芋/万年青/粗肋草 (33-40) ==========
    "Caladium bicolor (彩叶芋)",
    "Syngonium podophyllum (合果芋)",
    "Syngonium erythrophyllum (红叶合果芋)",
    "Dieffenbachia seguine (花叶万年青)",
    "Dieffenbachia maculata (黛粉叶)",
    "Aglaonema commutatum (银皇后)",
    "Aglaonema modestum (广东万年青)",
    "Aglaonema pictum (迷彩粗肋草)",

    # ========== 6. 竹芋科 (41-51) ==========
    "Calathea orbifolia (圆叶竹芋)",
    "Calathea ornata (双线竹芋)",
    "Calathea makoyana (孔雀竹芋)",
    "Calathea roseopicta (玫瑰竹芋)",
    "Calathea lancifolia (箭羽竹芋)",
    "Calathea zebrina (天鹅绒竹芋)",
    "Calathea warscewiczii (丝绒竹芋)",
    "Calathea musaica (网格竹芋)",
    "Maranta leuconeura (豹纹竹芋)",
    "Ctenanthe burle-marxii (锦竹芋)",
    "Stromanthe sanguinea (红背竹芋)",

    # ========== 7. 榕属 (52-60) ==========
    "Ficus elastica (橡皮树)",
    "Ficus lyrata (琴叶榕)",
    "Ficus benjamina (垂叶榕)",
    "Ficus microcarpa (榕树)",
    "Ficus pumila (薜荔)",
    "Ficus benghalensis (孟加拉榕)",
    "Ficus altissima (高山榕)",
    "Ficus religiosa (菩提树)",
    "Ficus triangularis (三角榕)",

    # ========== 8. 龙血树与朱蕉 (61-68) ==========
    "Dracaena marginata (龙血树)",
    "Dracaena fragrans (巴西木)",
    "Dracaena sanderiana (富贵竹)",
    "Dracaena reflexa (百合竹)",
    "Dracaena draco (龙血铁)",
    "Dracaena deremensis (银线龙血树)",
    "Cordyline fruticosa (朱蕉)",
    "Cordyline australis (新西兰朱蕉)",

    # ========== 9. 虎皮兰 (69-72) ==========
    "Sansevieria trifasciata (虎皮兰)",
    "Sansevieria cylindrica (棒叶虎皮兰)",
    "Sansevieria masoniana (鲸鱼鳍虎皮兰)",
    "Sansevieria kirkii (银虎皮兰)",

    # ========== 10. 椒草属 (73-78) ==========
    "Peperomia obtusifolia (豆瓣绿)",
    "Peperomia argyreia (西瓜皮椒草)",
    "Peperomia caperata (皱叶椒草)",
    "Peperomia polybotrya (荷叶椒草)",
    "Peperomia rotundifolia (圆叶椒草)",
    "Peperomia prostrata (佛珠椒草)",

    # ========== 11. 吊兰/吊竹梅/一叶兰 (79-84) ==========
    "Chlorophytum comosum (吊兰)",
    "Tradescantia zebrina (吊竹梅)",
    "Tradescantia fluminensis (白花紫露草)",
    "Tradescantia pallida (紫竹梅)",
    "Tradescantia spathacea (紫背万年青)",
    "Aspidistra elatior (一叶兰)",

    # ========== 12. 其他常见观叶 (85-109) ==========
    "Pachira aquatica (发财树)",
    "Schefflera actinophylla (鹅掌柴)",
    "Schefflera arboricola (鸭脚木)",
    "Zamioculcas zamiifolia (金钱树)",
    "Pilea peperomioides (镜面草)",
    "Pilea involucrata (冷水花)",
    "Codiaeum variegatum (变叶木)",
    "Fittonia verschaffeltii (网纹草)",
    "Hypoestes phyllostachya (嫣红蔓)",
    "Coleus scutellarioides (彩叶草)",
    "Fatsia japonica (八角金盘)",
    "Polyscias fruticosa (南洋参)",
    "Polyscias scutellaria (圆叶福禄桐)",
    "Yucca elephantipes (象脚丝兰)",
    "Beaucarnea recurvata (酒瓶兰)",
    "Pandanus utilis (露兜树)",
    "Asparagus setaceus (文竹)",
    "Asparagus densiflorus (天门冬)",
    "Clusia rosea (书带木)",
    "Strelitzia nicolai (大鹤望兰)",
    "Strelitzia reginae (鹤望兰)",
    "Heliconia rostrata (垂花蝎尾蕉)",
    "Ravenala madagascariensis (旅人蕉)",
    "Musa acuminata (观赏蕉)",
    "Alpinia zerumbet (艳山姜)",

    # ========== 13. 多肉植物 (110-164) ==========
    "Echeveria elegans (月影)",
    "Echeveria agavoides (东云)",
    "Echeveria lilacina (丁香石莲)",
    "Echeveria pulidonis (花月夜)",
    "Echeveria laui (雪莲)",
    "Echeveria colorata (卡罗拉)",
    "Echeveria peacockii (蓝石莲)",
    "Echeveria setosa (锦晃星)",
    "Echeveria derenbergii (静夜)",
    "Echeveria runyonii (特玉莲)",
    "Echeveria nodulosa (养老)",
    "Graptopetalum paraguayense (朦胧月)",
    "Graptoveria opalina (白牡丹)",
    "Graptosedum alpenglow (秋丽)",
    "Pachyphytum oviferum (桃美人)",
    "Pachyphytum compactum (千代田之松)",
    "Crassula ovata (玉树)",
    "Crassula perforata (星王子)",
    "Crassula muscosa (青锁龙)",
    "Crassula arborescens (银波锦)",
    "Crassula capitella (火祭)",
    "Aloe vera (芦荟)",
    "Aloe aristata (绫锦芦荟)",
    "Aloe brevifolia (短叶芦荟)",
    "Haworthia fasciata (条纹十二卷)",
    "Haworthia cooperi (玉露)",
    "Haworthia attenuata (十二卷)",
    "Haworthia retusa (寿)",
    "Haworthia truncata (玉扇)",
    "Gasteria carinata (鲨鱼掌)",
    "Sedum morganianum (翡翠景天)",
    "Sedum rubrotinctum (虹之玉)",
    "Sedum adolphii (黄丽)",
    "Sedum dasyphyllum (薄雪万年草)",
    "Sedum album (白景天)",
    "Sedum spurium (松叶景天)",
    "Kalanchoe tomentosa (月兔耳)",
    "Kalanchoe daigremontiana (落地生根)",
    "Kalanchoe beharensis (仙女之舞)",
    "Kalanchoe thyrsiflora (唐印)",
    "Lithops aucampiae (生石花)",
    "Sempervivum tectorum (长生草)",
    "Aeonium arboreum (莲花掌)",
    "Aeonium haworthii (爱染锦)",
    "Aeonium tabuliforme (明镜)",
    "Senecio rowleyanus (佛珠)",
    "Senecio radicans (弦月)",
    "Senecio herreianus (翡翠珠)",
    "Cotyledon tomentosa (熊童子)",
    "Adromischus cristatus (天锦章)",
    "Faucaria tigrina (虎颚花)",
    "Portulacaria afra (树马齿苋)",
    "Agave americana (龙舌兰)",
    "Agave victoriae-reginae (雷神)",
    "Agave attenuata (翠绿龙舌兰)",

    # ========== 14. 仙人掌 (165-184) ==========
    "Opuntia microdasys (金乌帽子)",
    "Mammillaria elongata (金手指)",
    "Mammillaria gracilis (银手指)",
    "Mammillaria plumosa (羽毛球)",
    "Mammillaria bocasana (高砂)",
    "Echinocactus grusonii (金琥)",
    "Gymnocalycium mihanovichii (绯牡丹)",
    "Astrophytum asterias (兜)",
    "Astrophytum myriostigma (鸾凤玉)",
    "Ferocactus latispinus (日之出丸)",
    "Cereus peruvianus (六角柱)",
    "Rhipsalis baccifera (丝苇)",
    "Rhipsalis cereuscula (珊瑚仙人掌)",
    "Schlumbergera truncata (蟹爪兰)",
    "Epiphyllum oxypetalum (昙花)",
    "Hylocereus undatus (火龙果)",
    "Selenicereus grandiflorus (量天尺)",
    "Echinopsis pachanoi (圣佩德罗)",
    "Notocactus leninghausii (金晃)",
    "Parodia magnifica (雪晃)",

    # ========== 15. 蕨类植物 (185-199) ==========
    "Nephrolepis exaltata (波士顿蕨)",
    "Nephrolepis cordifolia (肾蕨)",
    "Adiantum capillus-veneris (铁线蕨)",
    "Adiantum raddianum (梅花铁线蕨)",
    "Asplenium nidus (鸟巢蕨)",
    "Asplenium antiquum (山苏花)",
    "Platycerium bifurcatum (鹿角蕨)",
    "Platycerium superbum (巨大鹿角蕨)",
    "Platycerium coronarium (皇冠鹿角蕨)",
    "Pteris cretica (凤尾蕨)",
    "Davallia fejeensis (兔脚蕨)",
    "Selaginella kraussiana (卷柏)",
    "Selaginella uncinata (翠云草)",
    "Polypodium aureum (金毛蕨)",
    "Lygodium japonicum (海金沙)",

    # ========== 16. 兰科 (200-220) ==========
    "Phalaenopsis aphrodite (蝴蝶兰)",
    "Phalaenopsis amabilis (白蝴蝶兰)",
    "Phalaenopsis schilleriana (席勒蝴蝶兰)",
    "Dendrobium nobile (石斛兰)",
    "Dendrobium phalaenopsis (蝴蝶石斛)",
    "Dendrobium kingianum (粉石斛)",
    "Cymbidium sinense (墨兰)",
    "Cymbidium goeringii (春兰)",
    "Cymbidium ensifolium (建兰)",
    "Cymbidium faberi (蕙兰)",
    "Cymbidium kanran (寒兰)",
    "Oncidium flexuosum (文心兰)",
    "Cattleya labiata (嘉德利亚兰)",
    "Paphiopedilum insigne (兜兰)",
    "Vanda coerulea (万代兰)",
    "Miltonia spectabilis (堇花兰)",
    "Zygopetalum mackayi (紫香兰)",
    "Ludisia discolor (金线莲)",
    "Vanilla planifolia (香草兰)",
    "Brassia verrucosa (蜘蛛兰)",
    "Epidendrum radicans (树兰)",

    # ========== 17. 凤梨科 (221-234) ==========
    "Guzmania lingulata (擎天凤梨)",
    "Vriesea splendens (丹尼斯凤梨)",
    "Aechmea fasciata (蜻蜓凤梨)",
    "Neoregelia carolinae (彩叶凤梨)",
    "Tillandsia usneoides (松萝凤梨)",
    "Tillandsia ionantha (精灵空凤)",
    "Tillandsia xerographica (霸王空凤)",
    "Tillandsia caput-medusae (美杜莎空凤)",
    "Tillandsia bulbosa (球茎空凤)",
    "Tillandsia stricta (针叶空凤)",
    "Tillandsia cyanea (铁兰)",
    "Billbergia nutans (垂花水塔花)",
    "Ananas comosus (观赏菠萝)",
    "Cryptanthus bivittatus (姬凤梨)",

    # ========== 18. 秋海棠 (235-239) ==========
    "Begonia semperflorens (四季秋海棠)",
    "Begonia rex (蟆叶秋海棠)",
    "Begonia maculata (银点秋海棠)",
    "Begonia grandis (秋海棠)",
    "Begonia corallina (珊瑚秋海棠)",

    # ========== 19. 苦苣苔科 (240-248) ==========
    "Saintpaulia ionantha (非洲紫罗兰)",
    "Sinningia speciosa (大岩桐)",
    "Streptocarpus rexii (海豚花)",
    "Aeschynanthus radicans (口红花)",
    "Nematanthus gregarius (金鱼花)",
    "Episcia cupreata (喜荫花)",
    "Columnea gloriosa (哥伦比亚花)",
    "Kohleria eriantha (艳桐草)",
    "Achimenes longiflora (长筒花)",

    # ========== 20. 常见开花植物 (249-288) ==========
    "Rosa chinensis (月季)",
    "Gardenia jasminoides (栀子花)",
    "Jasminum sambac (茉莉花)",
    "Jasminum polyanthum (多花茉莉)",
    "Hibiscus rosa-sinensis (朱槿)",
    "Bougainvillea spectabilis (三角梅)",
    "Pelargonium hortorum (天竺葵)",
    "Pelargonium graveolens (香叶天竺葵)",
    "Hydrangea macrophylla (绣球花)",
    "Chrysanthemum morifolium (菊花)",
    "Lilium longiflorum (百合)",
    "Tulipa gesneriana (郁金香)",
    "Narcissus tazetta (水仙)",
    "Lavandula angustifolia (薰衣草)",
    "Cyclamen persicum (仙客来)",
    "Primula obconica (报春花)",
    "Clivia miniata (君子兰)",
    "Hippeastrum vittatum (朱顶红)",
    "Zantedeschia aethiopica (马蹄莲)",
    "Gerbera jamesonii (非洲菊)",
    "Eustoma grandiflorum (洋桔梗)",
    "Plumeria rubra (鸡蛋花)",
    "Adenium obesum (沙漠玫瑰)",
    "Euphorbia milii (虎刺梅)",
    "Euphorbia pulcherrima (一品红)",
    "Kalanchoe blossfeldiana (长寿花)",
    "Pentas lanceolata (五星花)",
    "Ixora chinensis (龙船花)",
    "Catharanthus roseus (长春花)",
    "Fuchsia hybrida (倒挂金钟)",
    "Impatiens walleriana (凤仙花)",
    "Impatiens hawkeri (新几内亚凤仙)",
    "Medinilla magnifica (宝莲花)",
    "Passiflora caerulea (西番莲)",
    "Hoya carnosa (球兰)",
    "Hoya kerrii (心叶球兰)",
    "Hoya pubicalyx (毛萼球兰)",
    "Hoya linearis (线叶球兰)",
    "Mandevilla sanderi (飘香藤)",
    "Stephanotis floribunda (非洲茉莉)",

    # ========== 21. 季节性开花 (289-307) ==========
    "Camellia japonica (山茶花)",
    "Rhododendron simsii (西洋杜鹃)",
    "Azalea indica (杜鹃花)",
    "Dahlia pinnata (大丽花)",
    "Dianthus caryophyllus (康乃馨)",
    "Dianthus chinensis (石竹)",
    "Ranunculus asiaticus (花毛茛)",
    "Helianthus annuus (向日葵)",
    "Tagetes erecta (万寿菊)",
    "Petunia hybrida (矮牵牛)",
    "Zinnia elegans (百日草)",
    "Cosmos bipinnatus (格桑花)",
    "Portulaca grandiflora (太阳花)",
    "Celosia argentea (鸡冠花)",
    "Gomphrena globosa (千日红)",
    "Gazania rigens (勋章菊)",
    "Osteospermum ecklonis (非洲雏菊)",
    "Anemone coronaria (银莲花)",
    "Helleborus niger (铁筷子)",

    # ========== 22. 棕榈科 (308-318) ==========
    "Chamaedorea elegans (袖珍椰子)",
    "Dypsis lutescens (散尾葵)",
    "Howea forsteriana (肯蒂棕)",
    "Rhapis excelsa (棕竹)",
    "Livistona chinensis (蒲葵)",
    "Phoenix roebelenii (矮枣椰)",
    "Caryota mitis (鱼尾葵)",
    "Chamaerops humilis (欧洲棕)",
    "Trachycarpus fortunei (棕榈)",
    "Areca catechu (槟榔)",
    "Cocos nucifera (椰子)",

    # ========== 23. 苏铁 (319-320) ==========
    "Cycas revoluta (苏铁)",
    "Zamia furfuracea (泽米铁)",

    # ========== 24. 攀援/垂吊植物 (321-328) ==========
    "Hedera helix (常春藤)",
    "Cissus rhombifolia (菱叶白粉藤)",
    "Ceropegia woodii (爱之蔓)",
    "Dischidia nummularia (纽扣玉)",
    "Thunbergia alata (翼叶山牵牛)",
    "Clematis florida (铁线莲)",
    "Trachelospermum jasminoides (络石)",
    "Ipomoea batatas (观赏红薯)",

    # ========== 25. 大戟属 (329-333) ==========
    "Euphorbia tirucalli (光棍树)",
    "Euphorbia lactea (帝锦)",
    "Euphorbia trigona (彩云阁)",
    "Euphorbia obesa (布纹球)",
    "Euphorbia ingens (大戟树)",

    # ========== 26. 芳香草本 (334-345) ==========
    "Mentha spicata (留兰香)",
    "Mentha piperita (薄荷)",
    "Ocimum basilicum (罗勒)",
    "Rosmarinus officinalis (迷迭香)",
    "Thymus vulgaris (百里香)",
    "Salvia officinalis (鼠尾草)",
    "Origanum vulgare (牛至)",
    "Petroselinum crispum (欧芹)",
    "Cymbopogon citratus (香茅)",
    "Stevia rebaudiana (甜菊)",
    "Melissa officinalis (蜜蜂花)",
    "Laurus nobilis (月桂)",

    # ========== 27. 食虫植物 (346-350) ==========
    "Dionaea muscipula (捕蝇草)",
    "Nepenthes alata (猪笼草)",
    "Sarracenia purpurea (瓶子草)",
    "Drosera capensis (毛毡苔)",
    "Pinguicula moranensis (捕虫堇)",

    # ========== 28. 水生/半水生 (351-357) ==========
    "Nelumbo nucifera (荷花)",
    "Nymphaea colorata (睡莲)",
    "Pistia stratiotes (大萍)",
    "Eichhornia crassipes (凤眼蓝)",
    "Cyperus alternifolius (风车草)",
    "Cyperus papyrus (纸莎草)",
    "Acorus gramineus (石菖蒲)",

    # ========== 29. 盆景/树木 (358-377) ==========
    "Podocarpus macrophyllus (罗汉松)",
    "Murraya paniculata (九里香)",
    "Serissa foetida (六月雪)",
    "Buxus sempervirens (黄杨)",
    "Carmona microphylla (福建茶)",
    "Ulmus parvifolia (榆树)",
    "Juniperus chinensis (圆柏)",
    "Pinus parviflora (五针松)",
    "Acer palmatum (红枫)",
    "Ginkgo biloba (银杏)",
    "Prunus serrulata (樱花)",
    "Wisteria sinensis (紫藤)",
    "Araucaria heterophylla (南洋杉)",
    "Olea europaea (橄榄树)",
    "Punica granatum (石榴)",
    "Ligustrum lucidum (女贞)",
    "Citrus reticulata (橘子)",
    "Citrus limon (柠檬)",
    "Citrus sinensis (甜橙)",
    "Fortunella margarita (金桔)",

    # ========== 30. 竹类 (378-381) ==========
    "Bambusa multiplex (凤尾竹)",
    "Bambusa ventricosa (佛肚竹)",
    "Phyllostachys aurea (金竹)",
    "Phyllostachys nigra (紫竹)",

    # ========== 31. 姜科 (382-385) ==========
    "Curcuma alismatifolia (姜荷花)",
    "Hedychium coronarium (姜花)",
    "Zingiber spectabile (观赏姜)",
    "Costus speciosus (闭鞘姜)",

    # ========== 32. 补充常见品种 (386-400) ==========
    "Oxalis triangularis (紫叶酢浆草)",
    "Oxalis regnellii (白花酢浆草)",
    "Mimosa pudica (含羞草)",
    "Nandina domestica (南天竹)",
    "Aucuba japonica (东瀛珊瑚)",
    "Dioscorea elephantipes (龟甲龙)",
    "Jatropha podagrica (佛肚树)",
    "Soleirolia soleirolii (婴儿泪)",
    "Homalomena rubescens (春雨)",
    "Tetrastigma voinierianum (崖爬藤)",
    "Ctenanthe oppenheimiana (彩虹竹芋)",
    "Calathea veitchiana (伟氏竹芋)",
    "Aegagropila linnaei (绿球藻)",
    "Brighamia insignis (夏威夷棕)",
    "Senecio macroglossus (蜡常春藤)",
    "Mangifera indica L(芒果)"
]


# =========================
# 3. 模型类
# =========================
class SpeciesPredictor:
    def __init__(self):
        if not os.path.exists(BIOCLIP_WEIGHT):
            raise FileNotFoundError(
                f"没有找到 BioCLIP 权重文件：{BIOCLIP_WEIGHT}\n"
                f"请把 open_clip_pytorch_model.bin 放到 model/ 目录下"
            )

        print("正在加载植物种类识别模型 BioCLIP...")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-B-16",
            pretrained=BIOCLIP_WEIGHT
        )

        self.tokenizer = open_clip.get_tokenizer("ViT-B-16")
        self.model.to(DEVICE)
        self.model.eval()

        self.species_list = PLANT_SPECIES
        self.scientific_names = [
            item.split("(")[0].strip()
            for item in self.species_list
        ]

        self.text_features = self.build_text_features()

        print("植物种类识别模型加载完成")

    def build_text_features(self):
        prompts = [
            f"a photo of {name}, a tropical plant species"
            for name in self.scientific_names
        ]

        with torch.no_grad():
            tokens = self.tokenizer(prompts).to(DEVICE)
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(
                dim=-1,
                keepdim=True
            )

        return text_features

    def predict(self, image: Image.Image, top_k=5):
        image = image.convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(
                dim=-1,
                keepdim=True
            )

            similarity = image_features @ self.text_features.T

            # 关键修改：温度缩放
            # 原来直接 softmax，400 类概率会被平均摊开，经常只有 0.3%
            # temperature 越小，Top1 概率越明显
            temperature = 0.03

            probs = (similarity / temperature).softmax(dim=-1)[0]

            top_probs, top_indices = torch.topk(probs, top_k)

        results = []

        for prob, idx in zip(top_probs, top_indices):
            results.append({
                "label": self.species_list[int(idx)],
                "confidence": float(prob)
            })

        return results


# =========================
# 4. 全局只加载一次模型
# =========================
_predictor = None


def get_predictor():
    global _predictor

    if _predictor is None:
        _predictor = SpeciesPredictor()

    return _predictor


# =========================
# 5. 给 app.py 调用的统一函数
# =========================
def predict_species(image: Image.Image):
    predictor = get_predictor()

    top5 = predictor.predict(image, top_k=5)

    best = top5[0]

    return {
        "species": best["label"],
        "confidence": best["confidence"],
        "top5": top5
    }