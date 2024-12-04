nohup python -u train_UTSRMorph_abdomen.py >> train_abdomen_mi.log 2>&1 &

nohup python -u train_UTSRMorph_abdomen_diceloss.py > train_abdomen_dice.log 2>&1 &


