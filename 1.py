import csv

# 完整实体数据
entities = [
    # 基本信息
    ["E1", "器官", "肝脏"],
    ["E2", "特性", "最大实质性器官"],
    ["E3", "特性", "最大腺体"],
    ["E4", "权重", "体重1/40-1/50"],
    
    # 解剖位置
    ["E5", "解剖位置", "腹腔右上区域"],
    ["E6", "解剖位置", "右侧横膈膜下方"],
    ["E7", "器官", "胆囊"],
    ["E8", "器官", "右肾"],
    ["E9", "器官", "胃"],
    ["E10", "解剖位置", "右侧第七至第十一根肋骨间"],
    
    # 生理特征
    ["E11", "属性", "重量_男性_1230-1450克"],
    ["E12", "属性", "重量_女性_1100-1300克"],
    ["E13", "属性", "颜色_棕红色"],
    ["E14", "属性", "血容量_14%"],
    ["E15", "属性", "血流量_1500-2000ml/min"],
    ["E16", "属性", "心输出量_25%"],
    
    # 解剖结构
    ["E17", "解剖系统", "奎诺系统"],
    ["E18", "解剖结构", "8个解剖节段"],
    ["E19", "解剖结构", "左叶"],
    ["E20", "解剖结构", "右叶"],
    ["E21", "解剖结构", "尾叶"],
    ["E22", "解剖结构", "方叶"],
    ["E23", "结构", "镰状韧带"],
    
    # 血液供应
    ["E24", "血管", "门静脉"],
    ["E25", "血管", "肝动脉"],
    ["E26", "属性", "门静脉供血_75%"],
    ["E27", "属性", "肝动脉供血_25%"],
    
    # 合成功能相关
    ["E28", "生物分子", "凝血因子I"],
    ["E29", "生物分子", "凝血因子II"],
    ["E30", "生物分子", "凝血因子V"],
    ["E31", "生物分子", "凝血因子VII"],
    ["E32", "生物分子", "凝血因子IX"],
    ["E33", "生物分子", "凝血因子XI"],
    ["E34", "蛋白质", "C反应蛋白"],
    ["E35", "蛋白质", "S反应蛋白"],
    ["E36", "蛋白质", "血浆蛋白"],
    ["E37", "生长因子", "胰岛素样生长因子1"],
    ["E38", "激素", "血小板生成素"],
    
    # 代谢功能相关
    ["E39", "代谢物质", "糖原"],
    ["E40", "代谢过程", "糖异生"],
    ["E41", "代谢物质", "胆固醇"],
    ["E42", "代谢物质", "甘油三酯"],
    ["E43", "代谢物质", "脂肪酸"],
    
    # 储存功能相关
    ["E44", "维生素", "维生素A"],
    ["E45", "维生素", "维生素D"],
    ["E46", "维生素", "维生素B12"],
    ["E47", "元素", "铁"],
    ["E48", "元素", "铜"],
    
    # 解毒功能相关
    ["E49", "酶系统", "P450系统"],
    ["E50", "生化过程", "氧化"],
    ["E51", "生化过程", "还原"],
    ["E52", "生化过程", "水解"],
    ["E53", "生化过程", "醛糖酸化"],
    ["E54", "生化过程", "硫酸化"],
    ["E55", "生化过程", "甘胺酸化"],
    ["E56", "生化过程", "谷胱甘肽结合"],
    
    # 疾病相关
    ["E57", "疾病", "病毒性肝炎"],
    ["E58", "疾病", "酒精性肝炎"],
    ["E59", "疾病", "自身免疫性肝炎"],
    ["E60", "疾病", "脂肪肝"],
    ["E61", "疾病", "威尔森氏症"],
    ["E62", "疾病", "血色沉着病"],
    ["E63", "疾病", "原发性肝癌"],
    ["E64", "疾病类型", "肝细胞癌"],
    ["E65", "疾病类型", "胆管细胞癌"],
    ["E66", "疾病", "转移性肝癌"],
    
    # 治疗相关
    ["E67", "治疗", "肝脏移植"],
    ["E68", "治疗类型", "活体肝移植"],
    ["E69", "治疗类型", "尸体肝移植"],
    ["E70", "研究方向", "人工肝研发"],
    ["E71", "研究方向", "干细胞治疗"],
    ["E72", "研究方向", "基因治疗"],
    ["E73", "研究方向", "新型药物研发"],
    ["E74", "研究方向", "微创手术技术"],
    
    # 症状相关
    ["E75", "症状", "黄疸"],
    ["E76", "症状", "腹水"],
    ["E77", "症状", "肝掌"],
    ["E78", "症状", "蜘蛛痣"],
    ["E79", "症状", "食欲不振"],
    ["E80", "症状", "乏力"],
    ["E81", "症状", "腹痛"],
    ["E82", "症状", "肝区压痛"],
    
    # 实验室检查
    ["E83", "检查指标", "ALT"],
    ["E84", "检查指标", "AST"],
    ["E85", "检查指标", "γ-GT"],
    ["E86", "检查指标", "ALP"],
    ["E87", "检查指标", "总胆红素"],
    ["E88", "检查指标", "白蛋白"],
    
    # 病理状态
    ["E89", "病理状态", "肝细胞坏死"],
    ["E90", "病理状态", "肝纤维化"],
    ["E91", "病理状态", "肝硬化"],
    ["E92", "病理状态", "门脉高压"],
    
    # 并发症
    ["E93", "并发症", "肝性脑病"],
    ["E94", "并发症", "食管胃底静脉曲张"],
    ["E95", "并发症", "肝肾综合征"],
    ["E96", "并发症", "自发性细菌性腹膜炎"],
    
    # 风险因素
    ["E97", "风险因素", "酗酒"],
    ["E98", "风险因素", "肥胖"],
    ["E99", "风险因素", "病毒感染"],
    ["E100", "风险因素", "药物性肝损伤"],
    
    # 生活方式
    ["E101", "生活方式", "戒酒"],
    ["E102", "生活方式", "低盐饮食"],
    ["E103", "生活方式", "规律作息"],
    ["E104", "生活方式", "适度运动"],
    
    # 诊断方法
    ["E105", "诊断方法", "肝脏超声"],
    ["E106", "诊断方法", "CT扫描"],
    ["E107", "诊断方法", "MRI检查"],
    ["E108", "诊断方法", "肝穿刺活检"],
    ["E109", "诊断方法", "血清学检查"],
    
    # 用药治疗
    ["E110", "药物", "抗病毒药物"],
    ["E111", "药物", "利尿剂"],
    ["E112", "药物", "肝保护剂"],
    ["E113", "药物", "免疫抑制剂"],
    
    # 预后指标
    ["E114", "预后指标", "Child-Pugh评分"],
    ["E115", "预后指标", "MELD评分"],
    ["E116", "预后指标", "腹水程度"],
    ["E117", "预后指标", "凝血功能"],
    
    # 护理措施
    ["E118", "护理措施", "营养支持"],
    ["E119", "护理措施", "皮肤护理"],
    ["E120", "护理措施", "并发症预防"],
    
    # 心理状态
    ["E121", "心理状态", "焦虑"],
    ["E122", "心理状态", "抑郁"],
    ["E123", "心理状态", "恐惧"],
    
    # 中医诊断
    ["E124", "中医证型", "肝郁气滞"],
    ["E125", "中医证型", "肝火上炎"],
    ["E126", "中医证型", "肝肾阴虚"],
    ["E127", "中医治疗", "中药汤剂"],
    ["E128", "中医治疗", "针灸治疗"],
    
    # 肝炎相关
    ["E129", "病毒类型", "甲型肝炎病毒"],
    ["E130", "病毒类型", "乙型肝炎病毒"],
    ["E131", "病毒类型", "丙型肝炎病毒"],
    ["E132", "病毒类型", "丁型肝炎病毒"],
    ["E133", "病毒类型", "戊型肝炎病毒"],
    ["E134", "预防", "疫苗接种"],
    ["E135", "检查", "病毒标志物"],
    ["E136", "检查", "HBV-DNA定量"],
    
    # 肝硬化相关
    ["E137", "分期", "代偿期"],
    ["E138", "分期", "失代偿期"],
    ["E139", "评分系统", "肝纤维化分期"],
    ["E140", "并发症", "上消化道出血"],
    ["E141", "并发症", "肝性胸水"],
    
    # 肝癌相关
    ["E142", "诊断指标", "AFP"],
    ["E143", "诊断指标", "AFP-L3"],
    ["E144", "分期系统", "BCLC分期"],
    ["E145", "治疗方案", "介入治疗"],
    ["E146", "治疗方案", "靶向治疗"],
    ["E147", "手术方式", "肝段切除"],
    ["E148", "手术方式", "肝叶切除"],
    
    # 药物性肝损伤
    ["E149", "损伤类型", "肝细胞损伤型"],
    ["E150", "损伤类型", "胆汁淤积型"],
    ["E151", "损伤类型", "混合型"],
    ["E152", "评分系统", "RUCAM评分"],
    
    # 病毒性肝炎治疗
    ["E153", "抗病毒药物", "恩替卡韦"],
    ["E154", "抗病毒药物", "替诺福韦"],
    ["E155", "抗病毒药物", "拉米夫定"],
    ["E156", "抗病毒药物", "干扰素"],
    ["E157", "抗病毒药物", "利巴韦林"],
    ["E158", "治疗方案", "联合抗病毒"],
    
    # 肝硬化治疗
    ["E159", "治疗药物", "β受体阻滞剂"],
    ["E160", "治疗药物", "螺内酯"],
    ["E161", "治疗方案", "内镜下套扎"],
    ["E162", "治疗方案", "内镜下硬化"],
    ["E163", "营养支持", "支链氨基酸"],
    ["E164", "营养支持", "白蛋白补充"],
    
    # 肝癌治疗
    ["E165", "手术方案", "肝移植评估"],
    ["E166", "介入治疗", "TACE"],
    ["E167", "介入治疗", "RFA"],
    ["E168", "靶向药物", "索拉菲尼"],
    ["E169", "靶向药物", "仑伐替尼"],
    
    # 自身免疫性肝病治疗
    ["E170", "治疗药物", "泼尼松"],
    ["E171", "治疗药物", "硫唑嘌呤"],
    ["E172", "治疗方案", "免疫抑制"],
    
    # 保肝治疗
    ["E173", "保肝药物", "水飞蓟素"],
    ["E174", "保肝药物", "还原型谷胱甘肽"],
    ["E175", "保肝药物", "甘草酸制剂"],
    ["E176", "中药制剂", "复方甘草酸苷"],
    ["E177", "中药制剂", "茵陈蒿汤"],
    
    # 生活质量评估
    ["E178", "生活质量", "睡眠质量"],
    ["E179", "生活质量", "工作能力"],
    ["E180", "生活质量", "社交活动"],
    
    # 并发症处理
    ["E181", "急救措施", "止血"],
    ["E182", "急救措施", "降氨"],
    ["E183", "护理措施", "伤口护理"],
    ["E184", "护理措施", "腹水抽排"],
    
    # 康复治疗
    ["E185", "康复措施", "运动康复"],
    ["E186", "康复措施", "饮食指导"],
    ["E187", "康复措施", "心理康复"],
    
    # 监测指标
    ["E188", "监测指标", "凝血酶原时间"],
    ["E189", "监测指标", "血氨"],
    ["E190", "监测指标", "电解质"],

    # 更多并发症
    ["E191", "并发症", "门脉血栓"],
    ["E192", "并发症", "肝性糖尿病"],
    ["E193", "并发症", "肝肺综合征"],
    ["E194", "并发症", "肝骨病"],
    ["E195", "并发症", "继发感染"],
    ["E196", "并发症", "肝性贫血"],
    ["E197", "并发症", "肝癌并发症"],
    ["E198", "并发症", "电解质紊乱"],
    ["E199", "并发症", "凝血功能障碍"],
    ["E200", "并发症", "免疫功能低下"],
    
    # 肝脏疾病分类
    ["E201", "疾病类别", "肝脏疾病"],
    ["E202", "疾病分类", "病毒性肝病"],
    ["E203", "疾病分类", "酒精性肝病"],
    ["E204", "疾病分类", "自身免疫性肝病"],
    ["E205", "疾病分类", "代谢性肝病"],
    ["E206", "疾病分类", "药物性肝病"],
    ["E207", "疾病分类", "肿瘤性肝病"],
    ["E208", "疾病特征", "急性"],
    ["E209", "疾病特征", "慢性"],
    ["E210", "疾病特征", "进行性"]
]

# 完整关系数据
relations = [
    # 基本特征关系
    ["E1", "是", "E2", "属性关系"],
    ["E1", "是", "E3", "属性关系"],
    ["E1", "占", "E4", "数量关系"],
    
    # 解剖位置关系
    ["E1", "位于", "E5", "位置关系"],
    ["E1", "位于", "E6", "位置关系"],
    ["E1", "位于前端", "E7", "位置关系"],
    ["E1", "位于前方", "E8", "位置关系"],
    ["E1", "位于上方", "E9", "位置关系"],
    ["E1", "跨越", "E10", "位置关系"],
    
    # 生理特征关系
    ["E1", "具有", "E11", "属性关系"],
    ["E1", "具有", "E12", "属性关系"],
    ["E1", "呈现", "E13", "属性关系"],
    ["E1", "具有", "E14", "属性关系"],
    ["E1", "具有", "E15", "属性关系"],
    ["E1", "接收", "E16", "属性关系"],
    
    # 结构关系
    ["E17", "划分", "E18", "分类关系"],
    ["E1", "包含", "E19", "组成关系"],
    ["E1", "包含", "E20", "组成关系"],
    ["E1", "包含", "E21", "组成关系"],
    ["E1", "包含", "E22", "组成关系"],
    ["E23", "分隔", "E19", "功能关系"],
    ["E23", "分隔", "E20", "功能关系"],
    
    # 血液供应关系
    ["E24", "提供", "E26", "供应关系"],
    ["E25", "提供", "E27", "供应关系"],
    
    # 合成功能关系
    ["E1", "合成", "E28", "功能关系"],
    ["E1", "合成", "E29", "功能关系"],
    ["E1", "合成", "E30", "功能关系"],
    ["E1", "合成", "E31", "功能关系"],
    ["E1", "合成", "E32", "功能关系"],
    ["E1", "合成", "E33", "功能关系"],
    ["E1", "合成", "E34", "功能关系"],
    ["E1", "合成", "E35", "功能关系"],
    ["E1", "合成", "E36", "功能关系"],
    ["E1", "合成", "E37", "功能关系"],
    ["E1", "合成", "E38", "功能关系"],
    
    # 代谢功能关系
    ["E1", "代谢", "E39", "功能关系"],
    ["E1", "执行", "E40", "功能关系"],
    ["E1", "合成", "E41", "功能关系"],
    ["E1", "生产", "E42", "功能关系"],
    ["E1", "代谢", "E43", "功能关系"],
    
    # 储存功能关系
    ["E1", "储存", "E44", "功能关系"],
    ["E1", "储存", "E45", "功能关系"],
    ["E1", "储存", "E46", "功能关系"],
    ["E1", "储存", "E47", "功能关系"],
    ["E1", "储存", "E48", "功能关系"],
    
    # 解毒功能关系
    ["E1", "依赖", "E49", "功能关系"],
    ["E49", "执行", "E50", "功能关系"],
    ["E49", "执行", "E51", "功能关系"],
    ["E49", "执行", "E52", "功能关系"],
    ["E1", "通过", "E53", "功能关系"],
    ["E1", "通过", "E54", "功能关系"],
    ["E1", "通过", "E55", "功能关系"],
    ["E1", "通过", "E56", "功能关系"],
    
    # 疾病关系
    ["E1", "可能患有", "E57", "疾病关系"],
    ["E1", "可能患有", "E58", "疾病关系"],
    ["E1", "可能患有", "E59", "疾病关系"],
    ["E1", "可能患有", "E60", "疾病关系"],
    ["E1", "可能患有", "E61", "疾病关系"],
    ["E1", "可能患有", "E62", "疾病关系"],
    ["E63", "包括", "E64", "分类关系"],
    ["E63", "包括", "E65", "分类关系"],
    ["E1", "可能患有", "E66", "疾病关系"],
    
    # 治疗关系
    ["E67", "包括", "E68", "分类关系"],
    ["E67", "包括", "E69", "分类关系"],
    ["E1", "相关研究", "E70", "研究关系"],
    ["E1", "相关研究", "E71", "研究关系"],
    ["E1", "相关研究", "E72", "研究关系"],
    ["E1", "相关研究", "E73", "研究关系"],
    ["E1", "相关研究", "E74", "研究关系"],
    
    # 症状关系
    ["E57", "表现为", "E75", "症状关系"],
    ["E91", "导致", "E76", "症状关系"],
    ["E91", "引起", "E77", "症状关系"],
    ["E91", "引起", "E78", "症状关系"],
    ["E57", "表现为", "E79", "症状关系"],
    ["E57", "表现为", "E80", "症状关系"],
    ["E63", "引起", "E81", "症状关系"],
    ["E57", "表现为", "E82", "症状关系"],
    
    # 检查指标关系
    ["E57", "升高", "E83", "检查关系"],
    ["E57", "升高", "E84", "检查关系"],
    ["E58", "升高", "E85", "检查关系"],
    ["E65", "升高", "E86", "检查关系"],
    ["E75", "相关", "E87", "检查关系"],
    ["E91", "降低", "E88", "检查关系"],
    
    # 病理进展关系
    ["E57", "导致", "E89", "病理关系"],
    ["E89", "进展为", "E90", "病理关系"],
    ["E90", "进展为", "E91", "病理关系"],
    ["E91", "引起", "E92", "病理关系"],
    
    # 并发症关系
    ["E91", "并发", "E93", "并发关系"],
    ["E92", "导致", "E94", "并发关系"],
    ["E91", "并发", "E95", "并发关系"],
    ["E91", "并发", "E96", "并发关系"],
    
    # 风险因素关系
    ["E97", "导致", "E58", "病因关系"],
    ["E98", "引起", "E60", "病因关系"],
    ["E99", "导致", "E57", "病因关系"],
    ["E100", "引起", "E89", "病因关系"],
    
    # 生活方式干预
    ["E101", "改善", "E58", "干预关系"],
    ["E102", "改善", "E76", "干预关系"],
    ["E103", "改善", "E80", "干预关系"],
    ["E104", "促进", "E60", "干预关系"],
    
    # 诊断方法关系
    ["E105", "诊断", "E60", "诊断关系"],
    ["E106", "诊断", "E63", "诊断关系"],
    ["E107", "诊断", "E66", "诊断关系"],
    ["E108", "确诊", "E59", "诊断关系"],
    ["E109", "检测", "E57", "诊断关系"],
    
    # 治疗关系
    ["E110", "治疗", "E57", "治疗关系"],
    ["E111", "治疗", "E76", "治疗关系"],
    ["E112", "保护", "E1", "治疗关系"],
    ["E113", "治疗", "E59", "治疗关系"],
    
    # 预后评估关系
    ["E114", "评估", "E91", "评估关系"],
    ["E115", "预测", "E67", "评估关系"],
    ["E116", "反映", "E76", "评估关系"],
    ["E117", "反映", "E91", "评估关系"],
    
    # 护理关系
    ["E118", "改善", "E79", "护理关系"],
    ["E119", "预防", "E78", "护理关系"],
    ["E120", "预防", "E93", "护理关系"],
    
    # 心理影响关系
    ["E57", "引起", "E121", "心理关系"],
    ["E91", "导致", "E122", "心理关系"],
    ["E63", "引起", "E123", "心理关系"],
    
    # 中医诊疗关系
    ["E124", "对应", "E57", "中医关系"],
    ["E125", "对应", "E58", "中医关系"],
    ["E126", "对应", "E59", "中医关系"],
    ["E127", "治疗", "E124", "治疗关系"],
    ["E128", "治疗", "E125", "治疗关系"],
    
    # 肝炎关系
    ["E129", "导致", "E57", "病因关系"],
    ["E130", "导致", "E57", "病因关系"],
    ["E131", "导致", "E57", "病因关系"],
    ["E132", "导致", "E57", "病因关系"],
    ["E133", "导致", "E57", "病因关系"],
    ["E134", "预防", "E130", "预防关系"],
    ["E135", "检测", "E130", "检查关系"],
    ["E136", "监测", "E130", "检查关系"],
    
    # 肝硬化关系
    ["E91", "分为", "E137", "分类关系"],
    ["E91", "分为", "E138", "分类关系"],
    ["E139", "评估", "E90", "评估关系"],
    ["E91", "并发", "E140", "并发关系"],
    ["E91", "并发", "E141", "并发关系"],
    
    # 肝癌关系
    ["E142", "诊断", "E63", "诊断关系"],
    ["E143", "诊断", "E63", "诊断关系"],
    ["E144", "评估", "E63", "评估关系"],
    ["E145", "治疗", "E63", "治疗关系"],
    ["E146", "治疗", "E63", "治疗关系"],
    ["E147", "切除", "E63", "治疗关系"],
    ["E148", "切除", "E63", "治疗关系"],
    
    # 药物性肝损伤关系
    ["E100", "包括", "E149", "分类关系"],
    ["E100", "包括", "E150", "分类关系"],
    ["E100", "包括", "E151", "分类关系"],
    ["E152", "评估", "E100", "评估关系"],
    
    # 病毒性肝炎治疗关系
    ["E153", "治疗", "E130", "治疗关系"],
    ["E154", "治疗", "E130", "治疗关系"],
    ["E155", "治疗", "E130", "治疗关系"],
    ["E156", "治疗", "E131", "治疗关系"],
    ["E157", "治疗", "E131", "治疗关系"],
    ["E158", "组合", "E156", "组合关系"],
    ["E158", "组合", "E157", "组合关系"],
    
    # 肝硬化治疗关系
    ["E159", "预防", "E140", "治疗关系"],
    ["E160", "治疗", "E76", "治疗关系"],
    ["E161", "治疗", "E94", "治疗关系"],
    ["E162", "治疗", "E94", "治疗关系"],
    ["E163", "改善", "E91", "治疗关系"],
    ["E164", "改善", "E91", "治疗关系"],
    
    # 肝癌治疗关系
    ["E165", "评估", "E67", "评估关系"],
    ["E166", "治疗", "E63", "治疗关系"],
    ["E167", "治疗", "E63", "治疗关系"],
    ["E168", "治疗", "E63", "治疗关系"],
    ["E169", "治疗", "E63", "治疗关系"],
    
    # 自身免疫性肝病治疗关系
    ["E170", "治疗", "E59", "治疗关系"],
    ["E171", "治疗", "E59", "治疗关系"],
    ["E172", "包括", "E170", "组合关系"],
    ["E172", "包括", "E171", "组合关系"],
    
    # 保肝治疗关系
    ["E173", "保护", "E1", "治疗关系"],
    ["E174", "保护", "E1", "治疗关系"],
    ["E175", "保护", "E1", "治疗关系"],
    ["E176", "保护", "E1", "治疗关系"],
    ["E177", "治疗", "E75", "治疗关系"],
    
    # 生活质量评估关系
    ["E91", "影响", "E178", "影响关系"],
    ["E91", "影响", "E179", "影响关系"],
    ["E91", "影响", "E180", "影响关系"],
    
    # 并发症处理关系
    ["E181", "处理", "E140", "处理关系"],
    ["E182", "处理", "E93", "处理关系"],
    ["E183", "处理", "E147", "处理关系"],
    ["E184", "处理", "E76", "处理关系"],
    
    # 康复治疗关系
    ["E185", "促进", "E1", "康复关系"],
    ["E186", "改善", "E79", "康复关系"],
    ["E187", "改善", "E121", "康复关系"],
    
    # 监测指标关系
    ["E188", "监测", "E91", "监测关系"],
    ["E189", "监测", "E93", "监测关系"],
    ["E190", "监测", "E95", "监测关系"],
    
    # 并发症关系
    ["E91", "并发", "E191", "并发关系"],
    ["E91", "并发", "E192", "并发关系"],
    ["E91", "并发", "E193", "并发关系"],
    ["E91", "并发", "E194", "并发关系"],
    ["E91", "并发", "E195", "并发关系"],
    ["E91", "并发", "E196", "并发关系"],
    ["E63", "并发", "E197", "并发关系"],
    ["E91", "并发", "E198", "并发关系"],
    ["E91", "并发", "E199", "并发关系"],
    ["E91", "导致", "E200", "并发关系"],
    
    # 肝脏疾病分类关系
    ["E201", "包含", "E202", "分类关系"],
    ["E201", "包含", "E203", "分类关系"],
    ["E201", "包含", "E204", "分类关系"],
    ["E201", "包含", "E205", "分类关系"],
    ["E201", "包含", "E206", "分类关系"],
    ["E201", "包含", "E207", "分类关系"],
    ["E202", "包含", "E57", "分类关系"],
    ["E203", "包含", "E58", "分类关系"],
    ["E204", "包含", "E59", "分类关系"],
    ["E206", "包含", "E100", "分类关系"],
    ["E207", "包含", "E63", "分类关系"],
    ["E207", "包含", "E66", "分类关系"],
    ["E208", "描述", "E57", "属性关系"],
    ["E209", "描述", "E91", "属性关系"],
    ["E210", "描述", "E90", "属性关系"]
]

# 写入实体CSV
with open('entities3.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['实体ID', '实体类型', '实体名称'])
    writer.writerows(entities)

# 写入关系CSV
with open('relations3.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['头实体ID', '关系类型', '尾实体ID', '关系类别'])
    writer.writerows(relations)