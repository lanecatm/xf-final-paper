





caseAttributeNameList = ['(case) amount_applied0','(case) amount_applied1','(case) amount_applied2','(case) amount_applied3','(case) applicant','(case) application','(case) area','(case) basic payment','(case) cross_compliance','(case) department','(case) greening','(case) number_parcels','(case) payment_actual0','(case) payment_actual1','(case) payment_actual2','(case) payment_actual3','(case) penalty_ABP','(case) penalty_AGP','(case) penalty_AJLP','(case) penalty_AUVP','(case) penalty_AVBP','(case) penalty_AVGP','(case) penalty_AVJLP','(case) penalty_AVUVP','(case) penalty_B16','(case) penalty_B2','(case) penalty_B3','(case) penalty_B4','(case) penalty_B5','(case) penalty_B5F','(case) penalty_B6','(case) penalty_BGK','(case) penalty_BGKV','(case) penalty_BGP','(case) penalty_C16','(case) penalty_C4','(case) penalty_C9','(case) penalty_CC','(case) penalty_GP1','(case) penalty_JLP1','(case) penalty_JLP2','(case) penalty_JLP3','(case) penalty_JLP5','(case) penalty_JLP6','(case) penalty_JLP7','(case) penalty_V5','(case) penalty_amount0','(case) penalty_amount1','(case) penalty_amount2','(case) penalty_amount3','(case) program-id','(case) redistribution','(case) rejected','(case) risk_factor','(case) selected_manually','(case) selected_random','(case) selected_risk','(case) small farmer','(case) year','(case) young farmer']
activityAttributeList = ['doctype','note','subprocess','success', "concept:name"]
defaultAtributeList = ['case id', 'Activity', 'Complete Timestamp', 'Resource']

useTimeAttributeList = ['Complete Timestamp']
useBooleanAttributeList = ['(case) basic payment', '(case) greening','(case) penalty_ABP','(case) penalty_AGP','(case) penalty_AJLP','(case) penalty_AUVP','(case) penalty_AVBP','(case) penalty_AVGP','(case) penalty_AVJLP','(case) penalty_AVUVP','(case) penalty_B16','(case) penalty_B2','(case) penalty_B3','(case) penalty_B4','(case) penalty_B5','(case) penalty_B5F','(case) penalty_B6','(case) penalty_BGK','(case) penalty_BGKV','(case) penalty_BGP','(case) penalty_C16','(case) penalty_C4','(case) penalty_C9','(case) penalty_CC','(case) penalty_GP1','(case) penalty_JLP1','(case) penalty_JLP2','(case) penalty_JLP3','(case) penalty_JLP5','(case) penalty_JLP6','(case) penalty_JLP7','(case) penalty_V5','(case) redistribution', '(case) rejected','(case) selected_manually','(case) selected_random', '(case) selected_risk', '(case) small farmer', '(case) young farmer']
useFloatAttributeList = ['(case) amount_applied0', '(case) amount_applied1', '(case) amount_applied2', '(case) amount_applied3', '(case) area', '(case) cross_compliance', '(case) number_parcels', '(case) payment_actual0','(case) payment_actual1','(case) payment_actual2','(case) payment_actual3','(case) penalty_amount0','(case) penalty_amount1','(case) penalty_amount2','(case) penalty_amount3', '(case) risk_factor']
useClassAttributeList = ['Activity', '(case) department', '(case) year', 'Resource']


caseActivityDict, vocabulary, timeOrderEventsArray, timeOrderLabelArray = load_data_from_db(num_steps = num_steps, defaultAtributeList= defaultAtributeList, activityAttributeList = activityAttributeList
        , caseAttributeNameList = caseAttributeNameList, useTimeAttributeList = useTimeAttributeList, useBooleanAttributeList = useBooleanAttributeList, useFloatAttributeList = useFloatAttributeList, useClassAttributeList = useClassAttributeList, caseColumnName = "case id", timeColumnName = "Complete Timestamp", idColumnName = "ID", dbName = "bpi2018.db", tableName = "BPIC2018_new", timeStrp = "%Y/%m/%d %H:%M:%S")
