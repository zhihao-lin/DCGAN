# python3 main.py train g_01 d_01 -id 001
# python3 main.py train g_02 d_02 -id 002
# python3 main.py train g_02 d_02 -id 003
# python3 main.py train g_tutorial d_tutorial -id reproduce 
# python3 main.py train g_tutorial d_tutorial -id exp1 -info "without initialization"
# python3 main.py train g_tutorial d_tutorial -id exp2 -info "change learning rate" -lr 1e-4
# python3 main.py train g_tutorial d_tutorial -id exp3 -info "change learning rate to 1e-2" -lr 1e-2
# python3 main.py train g_tutorial d_tutorial -id exp4 -info "use default beta value for Adam" -beta 0.9
python3 main.py train g_tutorial d_tutorial -id "exp5" -info "set bata_1 to 0"
