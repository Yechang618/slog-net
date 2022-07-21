def get_modelList(graphType, list_number):
    model_list = {}
    if graphType == 'Random Geometric':
        if list_number == 0:
            ## Experiments of training with the original noisy data
            model_list['modelDirList_0'] = ['sourceLocSLOGNET-Random Geometric-20220626193044',
               'sourceLocSLOGNET-Random Geometric-20220626193822',
                'sourceLocSLOGNET-Random Geometric-20220626201036',
                'sourceLocSLOGNET-Random Geometric-20220626204203',
                'sourceLocSLOGNET-Random Geometric-20220626221850',
                'sourceLocSLOGNET-Random Geometric-20220626225441',
                'sourceLocSLOGNET-Random Geometric-20220626232753',
                'sourceLocSLOGNET-Random Geometric-20220626225441',
                'sourceLocSLOGNET-Random Geometric-20220626232753',
                'sourceLocSLOGNET-Random Geometric-20220627000028'
               ]

            model_list['modelDirList_1'] = ['sourceLocSLOGNET-Random Geometric-20220627003548',
                 'sourceLocSLOGNET-Random Geometric-20220627011123',
                 'sourceLocSLOGNET-Random Geometric-20220627014519',
                 'sourceLocSLOGNET-Random Geometric-20220627021813',
                  'sourceLocSLOGNET-Random Geometric-20220627025343',
                  'sourceLocSLOGNET-Random Geometric-20220627032731',
                  'sourceLocSLOGNET-Random Geometric-20220627040029',
                  'sourceLocSLOGNET-Random Geometric-20220627043400',
                  'sourceLocSLOGNET-Random Geometric-20220627050606',
                  'sourceLocSLOGNET-Random Geometric-20220627053850'
                  ]

            model_list['modelDirList_2'] = ['sourceLocSLOGNET-Random Geometric-20220627061155',
                  'sourceLocSLOGNET-Random Geometric-20220627064820',
                  'sourceLocSLOGNET-Random Geometric-20220627072712',
                  'sourceLocSLOGNET-Random Geometric-20220627080026',
                  'sourceLocSLOGNET-Random Geometric-20220627083325',
                  'sourceLocSLOGNET-Random Geometric-20220627090703',
                  'sourceLocSLOGNET-Random Geometric-20220627094138',
                  'sourceLocSLOGNET-Random Geometric-20220627101545',
                  'sourceLocSLOGNET-Random Geometric-20220627105022',
                  'sourceLocSLOGNET-Random Geometric-20220627112317'
                 ]
    elif graphType == 'ER':
        if list_number == 0:
            ## Experiments of training with the original noisy data
            model_list['modelDirList_0'] = ['sourceLocSLOGNET-ER-20220627180032',
                  'sourceLocSLOGNET-ER-20220627180233',
                  'sourceLocSLOGNET-ER-20220627183452',
                  'sourceLocSLOGNET-ER-20220627190900',
                  'sourceLocSLOGNET-ER-20220627194225',
                  'sourceLocSLOGNET-ER-20220627201456',
                  'sourceLocSLOGNET-ER-20220627204704',
                  'sourceLocSLOGNET-ER-20220627211942',
                  'sourceLocSLOGNET-ER-20220627215415',
                  'sourceLocSLOGNET-ER-20220627222703'
                ]
            model_list['modelDirList_1'] = ['sourceLocSLOGNET-ER-20220627233707',
                  'sourceLocSLOGNET-ER-20220628001009',
                  'sourceLocSLOGNET-ER-20220628004336',
                  'sourceLocSLOGNET-ER-20220628011705',
                  'sourceLocSLOGNET-ER-20220628014954',
                  'sourceLocSLOGNET-ER-20220628022340',
                  'sourceLocSLOGNET-ER-20220628025804',
                  'sourceLocSLOGNET-ER-20220628033130',
                  'sourceLocSLOGNET-ER-20220628040621',
                  'sourceLocSLOGNET-ER-20220628044101'
                 ]

            model_list['modelDirList_2'] = ['sourceLocSLOGNET-ER-20220628051822',
                  'sourceLocSLOGNET-ER-20220628055436',
                  'sourceLocSLOGNET-ER-20220628063125',
                  'sourceLocSLOGNET-ER-20220628070942',
                  'sourceLocSLOGNET-ER-20220628074424',
                  'sourceLocSLOGNET-ER-20220628081937',
                  'sourceLocSLOGNET-ER-20220628085328',
                  'sourceLocSLOGNET-ER-20220628092850',
                  'sourceLocSLOGNET-ER-20220628100637',
                  'sourceLocSLOGNET-ER-20220628104252'
                ]
        elif list_number == 1:
            model_list['modelDirList_0'] = ['sourceLocSLOGNET-ER-20220629214454',
                                            'sourceLocSLOGNET-ER-20220629221905',
                                            'sourceLocSLOGNET-ER-20220629225421',
                                            'sourceLocSLOGNET-ER-20220629232740',
                                            'sourceLocSLOGNET-ER-20220630000148',
                                            'sourceLocSLOGNET-ER-20220630003641',
                                            'sourceLocSLOGNET-ER-20220630011052',
                                            'sourceLocSLOGNET-ER-20220630014734',
                                            'sourceLocSLOGNET-ER-20220630022320',
                                            'sourceLocSLOGNET-ER-20220630025827'
                                             ]
            model_list['modelDirList_0_label'] = ['Noise level 0']
            model_list['modelDirList_1'] = ['sourceLocSLOGNET-ER-20220630033252',
                                            'sourceLocSLOGNET-ER-20220630040724',
                                            'sourceLocSLOGNET-ER-20220630044200',
                                            'sourceLocSLOGNET-ER-20220630051630',
                                            'sourceLocSLOGNET-ER-20220630055025',
                                            'sourceLocSLOGNET-ER-20220630062501',
                                            'sourceLocSLOGNET-ER-20220630065953',
                                            'sourceLocSLOGNET-ER-20220630073529',
                                            'sourceLocSLOGNET-ER-20220630081113',
                                            'sourceLocSLOGNET-ER-20220630084750'                                               ]         
            model_list['modelDirList_1_label'] = ['Noise level 0.02']
            model_list['modelDirList_2'] = ['sourceLocSLOGNET-ER-20220630092259',
                                            'sourceLocSLOGNET-ER-20220630095758',
                                            'sourceLocSLOGNET-ER-20220630103151',
                                            'sourceLocSLOGNET-ER-20220630110623',
                                            'sourceLocSLOGNET-ER-20220630114015',
                                            'sourceLocSLOGNET-ER-20220630121407',
                                            'sourceLocSLOGNET-ER-20220630124846',
                                            'sourceLocSLOGNET-ER-20220630132258',
                                            'sourceLocSLOGNET-ER-20220630135744',
                                            'sourceLocSLOGNET-ER-20220630143254'
                                           ]
            model_list['modelDirList_2_label'] = ['Noise level 0.05']
            
        elif list_number == 2:
            model_list['modelDirList_0'] = ['sourceLocSLOGNET-ER-20220630190809',
                                            'sourceLocSLOGNET-ER-20220630191852',
                                            'sourceLocSLOGNET-ER-20220630192325',
                                            'sourceLocSLOGNET-ER-20220630192757',
                                            'sourceLocSLOGNET-ER-20220630193231',
                                            'sourceLocSLOGNET-ER-20220630193707',
                                            'sourceLocSLOGNET-ER-20220630194139',
                                            'sourceLocSLOGNET-ER-20220630194619',
                                            'sourceLocSLOGNET-ER-20220630195102',
                                            'sourceLocSLOGNET-ER-20220630195546'
            ]
            model_list['modelDirList_0_label'] = ['Noise level 0']
            model_list['modelDirList_1'] = ['sourceLocSLOGNET-ER-20220630200030',
                                            'sourceLocSLOGNET-ER-20220630200513',
                                            'sourceLocSLOGNET-ER-20220630200957',
                                            'sourceLocSLOGNET-ER-20220630201442',
                                            'sourceLocSLOGNET-ER-20220630201917',
                                            'sourceLocSLOGNET-ER-20220630202352',
                                            'sourceLocSLOGNET-ER-20220630202827',
                                            'sourceLocSLOGNET-ER-20220630203305',
                                            'sourceLocSLOGNET-ER-20220630203739',
                                            'sourceLocSLOGNET-ER-20220630204216'
            ]
            model_list['modelDirList_1_label'] = ['Noise level 0.02']
            model_list['modelDirList_2'] = ['sourceLocSLOGNET-ER-20220630204657',
                                            'sourceLocSLOGNET-ER-20220630205132',
                                            'sourceLocSLOGNET-ER-20220630205607',
                                            'sourceLocSLOGNET-ER-20220630210040',
                                            'sourceLocSLOGNET-ER-20220630210519',
                                            'sourceLocSLOGNET-ER-20220630210952',
                                            'sourceLocSLOGNET-ER-20220630211426',
                                            'sourceLocSLOGNET-ER-20220630211901',
                                            'sourceLocSLOGNET-ER-20220630212336',
                                            'sourceLocSLOGNET-ER-20220630212811'
                
                                           ]
            model_list['modelDirList_2_label'] = ['Noise level 0.05']            
            
        elif list_number == 3:
            # Experiment for N_realiz = 200
            model_list['modelDirList_0'] = ['sourceLocSLOGNET-ER-20220701194949',
                                            'sourceLocSLOGNET-ER-20220701210924',
                                            'sourceLocSLOGNET-ER-20220701222835',
                                            'sourceLocSLOGNET-ER-20220701233415',
                                            'sourceLocSLOGNET-ER-20220702005447',
                                            'sourceLocSLOGNET-ER-20220702021508',
                                            'sourceLocSLOGNET-ER-20220702033741',
                                            'sourceLocSLOGNET-ER-20220702050327',
                                            'sourceLocSLOGNET-ER-20220702062749',
                                            'sourceLocSLOGNET-ER-20220702075347'
           ]
            model_list['modelDirList_0_label'] = ['Noise level 0, SLOG-net v3, q = 4, trainmode = g']
            model_list['modelDirList_1'] = ['sourceLocSLOGNET-ER-20220702091806',
                                            'sourceLocSLOGNET-ER-20220702103358',
                                            'sourceLocSLOGNET-ER-20220702115336',
                                            'sourceLocSLOGNET-ER-20220702130812',
                                            'sourceLocSLOGNET-ER-20220702142324',
                                            'sourceLocSLOGNET-ER-20220702153949',
                                            'sourceLocSLOGNET-ER-20220702203135',
                                            'sourceLocSLOGNET-ER-20220702181223',
                                            'sourceLocSLOGNET-ER-20220702192514',
                                            'sourceLocSLOGNET-ER-20220702165543'
                                            ]
            model_list['modelDirList_1_label'] = ['Noise level 0, SLOG-net v1']
            
            model_list['modelDirList_2'] = ['sourceLocSLOGNET-ER-20220702201521',
                                           'sourceLocSLOGNET-ER-20220702190912',
                                           'sourceLocSLOGNET-ER-20220702212038',
                                           'sourceLocSLOGNET-ER-20220702223844',
                                           'sourceLocSLOGNET-ER-20220703000400',
                                           'sourceLocSLOGNET-ER-20220703012811',
                                           'sourceLocSLOGNET-ER-20220703024627',
                                           'sourceLocSLOGNET-ER-20220703040628',
                                           'sourceLocSLOGNET-ER-20220703065216',
                                           'sourceLocSLOGNET-ER-20220703052853']
            model_list['modelDirList_2_label'] = ['Noise level 0, SLOG-net v3, q = 4, trainmode = h']  
            
            
            model_list['modelDirList_3'] = ['sourceLocSLOGNET-ER-20220703081727',
                                           'sourceLocSLOGNET-ER-20220703110540',
                                           'sourceLocSLOGNET-ER-20220703094248',
                                           'sourceLocSLOGNET-ER-20220703123000',
                                           'sourceLocSLOGNET-ER-20220703135640',
                                           'sourceLocSLOGNET-ER-20220703152216',
                                           'sourceLocSLOGNET-ER-20220703164651',
                                           'sourceLocSLOGNET-ER-20220703180736',
                                           'sourceLocSLOGNET-ER-20220703193038',
                                           'sourceLocSLOGNET-ER-20220703204113']
            model_list['modelDirList_3_label'] = ['Noise level 0, SLOG-net v3, q = 10, trainmode = g'] 
            
            model_list['modelDirList_4'] = ['sourceLocSLOGNET-ER-20220703191631',
                                           'sourceLocSLOGNET-ER-20220703202751',
                                           'sourceLocSLOGNET-ER-20220703213450',
                                           'sourceLocSLOGNET-ER-20220703225544',
                                           'sourceLocSLOGNET-ER-20220704002219',
                                           'sourceLocSLOGNET-ER-20220704014740',
                                           'sourceLocSLOGNET-ER-20220704031546',
                                           'sourceLocSLOGNET-ER-20220704044207',
                                           'sourceLocSLOGNET-ER-20220704060743',
                                           'sourceLocSLOGNET-ER-20220704073623']
                                           
            model_list['modelDirList_4_label'] = ['Noise level 0, SLOG-net v3, q = 10, trainmode = h']   
            
            
            model_list['modelDirList_5'] = ['sourceLocSLOGNET-ER-20220704090445',
                                           'sourceLocSLOGNET-ER-20220704103019',
                                           'sourceLocSLOGNET-ER-20220704115356',
                                           'sourceLocSLOGNET-ER-20220704131831',
                                           'sourceLocSLOGNET-ER-20220704144319']   
            
            model_list['modelDirList_5_label'] = ['Noise level 0, SLOG-net v1, trainmode = h']     
            
            model_list['modelDirList_6'] = ['sourceLocSLOGNET-ER-20220704173424',
                                           'sourceLocSLOGNET-ER-20220704185849',
                                           'sourceLocSLOGNET-ER-20220704202203',
                                           'sourceLocSLOGNET-ER-20220704214621',
                                           'sourceLocSLOGNET-ER-20220704230643',
                                           'sourceLocSLOGNET-ER-20220705003423',
                                           'sourceLocSLOGNET-ER-20220705015649',
                                           'sourceLocSLOGNET-ER-20220705032510',
                                           'sourceLocSLOGNET-ER-20220705045404',
                                           'sourceLocSLOGNET-ER-20220705061505']   
            
            model_list['modelDirList_6_label'] = ['Noise level 0.02, SLOG-net v3, q = 4, trainmode = h']   
            
            model_list['modelDirList_7'] = ['sourceLocSLOGNET-ER-20220705074430',
                                           'sourceLocSLOGNET-ER-20220705091028',
                                           'sourceLocSLOGNET-ER-20220705103958',
                                           'sourceLocSLOGNET-ER-20220705120147',
                                           'sourceLocSLOGNET-ER-20220705132854',
                                           'sourceLocSLOGNET-ER-20220705144915',
                                           'sourceLocSLOGNET-ER-20220705161440',
                                           'sourceLocSLOGNET-ER-20220705174312',
                                           'sourceLocSLOGNET-ER-20220705191337',
                                           'sourceLocSLOGNET-ER-20220705202940'
                                           
                                           ]   
            
            model_list['modelDirList_7_label'] = ['Noise level 0.05, SLOG-net v3, q = 4, trainmode = h']     
            
            model_list['modelDirList_8'] = ['sourceLocSLOGNET-ER-20220705194316',
                                           'sourceLocSLOGNET-ER-20220705205145',
                                           'sourceLocSLOGNET-ER-20220705220303',
                                           'sourceLocSLOGNET-ER-20220705232839',
                                           'sourceLocSLOGNET-ER-20220706004924',
                                           'sourceLocSLOGNET-ER-20220706021308',
                                           'sourceLocSLOGNET-ER-20220706033249',
                                           'sourceLocSLOGNET-ER-20220706045448',
                                           'sourceLocSLOGNET-ER-20220706055115',
                                           'sourceLocSLOGNET-ER-20220706071741']   
            
            model_list['modelDirList_8_label'] = ['Noise level 0, SLOG-net v3, q = 1, trainmode = h']  
            
            model_list['modelDirList_9'] = ['sourceLocSLOGNET-ER-20220706083644',
                                           'sourceLocSLOGNET-ER-20220706095922',
                                           'sourceLocSLOGNET-ER-20220706112256',
                                           'sourceLocSLOGNET-ER-20220706125051',
                                           'sourceLocSLOGNET-ER-20220706141454',
                                           'sourceLocSLOGNET-ER-20220706153931',
                                           'sourceLocSLOGNET-ER-20220706170534',
                                           'sourceLocSLOGNET-ER-20220706182941',
                                           'sourceLocSLOGNET-ER-20220706194343',
                                           'sourceLocSLOGNET-ER-20220706205341'
                                           ]   
            
            model_list['modelDirList_9_label'] = ['Noise level 0, SLOG-net v3, q = 20, trainmode = h'] 
                                            
            model_list['modelDirList_10'] = ['sourceLocSLOGNET-ER-20220706175730',
                                           'sourceLocSLOGNET-ER-20220706191146',
                                           'sourceLocSLOGNET-ER-20220706202244',
                                           'sourceLocSLOGNET-ER-20220706212948',
                                           'sourceLocSLOGNET-ER-20220706224531',
                                           'sourceLocSLOGNET-ER-20220706235821',
                                           'sourceLocSLOGNET-ER-20220707012505',
                                           'sourceLocSLOGNET-ER-20220707030002',
                                           'sourceLocSLOGNET-ER-20220707042750',
                                           'sourceLocSLOGNET-ER-20220707055336']   
            
            model_list['modelDirList_10_label'] = ['Noise level 0, SLOG-net v3, q = 2, trainmode = h']                                             
    elif graphType == 'SBM':              
        if list_number == 0:  
            model_list['modelDirList_test'] = ['sourceLocSLOGNET-SBM-20220708195206']
            model_list['modelDirList_test_label'] = ['Noise level 0, SLOG-net v3 (q = 4) and CrdGNN, trainmode = h']   
            
            
            model_list['modelDirList_0'] = ['sourceLocSLOGNET-SBM-20220707213953',
                                           'sourceLocSLOGNET-SBM-20220707220500',
                                           'sourceLocSLOGNET-SBM-20220707223046',
                                           'sourceLocSLOGNET-SBM-20220707225742',
                                           'sourceLocSLOGNET-SBM-20220707232408',
                                           'sourceLocSLOGNET-SBM-20220707234905',
                                           'sourceLocSLOGNET-SBM-20220708001612',
                                           'sourceLocSLOGNET-SBM-20220708004242',
                                           'sourceLocSLOGNET-SBM-20220708010801',
                                           'sourceLocSLOGNET-SBM-20220708013259']   
            
            model_list['modelDirList_0_label'] = ['Noise level 0, SLOG-net v3 (q = 4) and CrdGNN, trainmode = h (100k)']  
           
            model_list['modelDirList_1'] = ['sourceLocSLOGNET-SBM-20220708200533',
                                           'sourceLocSLOGNET-SBM-20220708205600',
                                            'sourceLocSLOGNET-SBM-20220708214539',
                                            'sourceLocSLOGNET-SBM-20220708223459',
                                           'sourceLocSLOGNET-SBM-20220708232458',
                                           'sourceLocSLOGNET-SBM-20220709001624',
                                           'sourceLocSLOGNET-SBM-20220709010631',
                                           'sourceLocSLOGNET-SBM-20220709015624',
                                           'sourceLocSLOGNET-SBM-20220709024602',
                                           'sourceLocSLOGNET-SBM-20220709033619']   
            
            model_list['modelDirList_1_label'] = ['Noise level 0, SLOG-net v3 (q = 4) and CrdGNN, trainmode = h(200k)']     
            
        elif list_number == 1:  
            # List test
            model_list['modelDirList_test'] = ['sourceLocSLOGNET-SBM-20220710021822']
            model_list['modelDirList_test_label'] = ['Noise level 0, SLOG-net v3 (q = 4) and CrdGNN, trainmode = h (test)']   
            
            # List 0
            model_list['modelDirList_0'] = ['sourceLocSLOGNET-SBM-20220710024729',
                                           'sourceLocSLOGNET-SBM-20220710025005',
                                           'sourceLocSLOGNET-SBM-20220710025242',
                                           'sourceLocSLOGNET-SBM-20220710025519',
                                           'sourceLocSLOGNET-SBM-20220710025757',
                                           'sourceLocSLOGNET-SBM-20220710030035',
                                           'sourceLocSLOGNET-SBM-20220710030314',
                                           'sourceLocSLOGNET-SBM-20220710030556',
                                           'sourceLocSLOGNET-SBM-20220710030838',
                                           'sourceLocSLOGNET-SBM-20220710031120']   
            model_list['modelDirList_0_label'] = ['Noise level 0, SLOG-net v3 (q = 4) and CrdGNN, trainmode = h (50k, run at home)']  
            # List 1
            model_list['modelDirList_1'] = ['sourceLocSLOGNET-SBM-20220710111119',
                                           'sourceLocSLOGNET-SBM-20220710111708',
                                           'sourceLocSLOGNET-SBM-20220710112406',
                                           'sourceLocSLOGNET-SBM-20220710113143',
                                           'sourceLocSLOGNET-SBM-20220710113908',
                                           'sourceLocSLOGNET-SBM-20220710114658',
                                           'sourceLocSLOGNET-SBM-20220710115331',
                                           'sourceLocSLOGNET-SBM-20220710115936',
                                           'sourceLocSLOGNET-SBM-20220710120520',
                                           'sourceLocSLOGNET-SBM-20220710121110']   
            model_list['modelDirList_1_label'] = ['Noise level 0, SLOG-net v3 (q = 4) and CrdGNN, trainmode = h(100k, run at home)']    
            
            # List 2
            model_list['modelDirList_2'] = ['sourceLocSLOGNET-SBM-20220710160227',
                                           'sourceLocSLOGNET-SBM-20220710171540',
#                                            'sourceLocSLOGNET-SBM-20220710183328',  # The class seperation is problematic
                                           'sourceLocSLOGNET-SBM-20220710195059',
                                           'sourceLocSLOGNET-SBM-20220710210552',
                                            'sourceLocSLOGNET-SBM-20220710221824',
                                            'sourceLocSLOGNET-SBM-20220710233252',
                                           'sourceLocSLOGNET-SBM-20220711004525',
                                           'sourceLocSLOGNET-SBM-20220711015410',
                                           'sourceLocSLOGNET-SBM-20220711030816']   
            model_list['modelDirList_2_label'] = ['Noise level 0, SLOG-net v3 (q = 4) and CrdGNN, trainmode = h(200k)']    
            
            model_list['modelDirList_3'] = ['sourceLocSLOGNET-SBM-20220711001703',
                                           'sourceLocSLOGNET-SBM-20220711002911',
                                            'sourceLocSLOGNET-SBM-20220711004126',
                                            'sourceLocSLOGNET-SBM-20220711005418',
                                            'sourceLocSLOGNET-SBM-20220711010645',
                                            'sourceLocSLOGNET-SBM-20220711011900',
                                            'sourceLocSLOGNET-SBM-20220711013018',
                                            'sourceLocSLOGNET-SBM-20220711014209',
                                            'sourceLocSLOGNET-SBM-20220711015425',
                                            'sourceLocSLOGNET-SBM-20220711020626'
                                           ]   
            model_list['modelDirList_3_label'] = ['Noise level 0, SLOG-net v3 (q = 4) and CrdGNN, trainmode = h(200k, run at home)']              
            
            model_list['modelDirList_4'] = ['sourceLocSLOGNET-SBM-20220711135027',
                                           'sourceLocSLOGNET-SBM-20220711140048',
                                            'sourceLocSLOGNET-SBM-20220711141132',
                                           'sourceLocSLOGNET-SBM-20220711142205',
                                            'sourceLocSLOGNET-SBM-20220711143256',
                                            'sourceLocSLOGNET-SBM-20220711144346',
                                            'sourceLocSLOGNET-SBM-20220711145438',
                                            'sourceLocSLOGNET-SBM-20220711150534'
                                           ]   
            model_list['modelDirList_4_label'] = ['Noise level 0.02, SLOG-net v3 (q = 4) and CrdGNN, trainmode = h(100k, run at home)']      
            
            model_list['modelDirList_5'] = ['sourceLocSLOGNET-SBM-20220711190053',
                                            'sourceLocSLOGNET-SBM-20220711190441',
                                            'sourceLocSLOGNET-SBM-20220711190824',
                                            'sourceLocSLOGNET-SBM-20220711191209',
                                            'sourceLocSLOGNET-SBM-20220711191554'
                                           ]   
            model_list['modelDirList_5_label'] = ['Noise level 0.0, SLOG-net v3 (q = 4) and CrdGNN, filter = wt(100k, run at office)']                 
    return model_list
        