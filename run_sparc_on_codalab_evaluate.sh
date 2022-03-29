pip3 install sqlparse
pip3 install nltk
python nltk_downloader.py

echo "ls and pwd"
ls
pwd

echo "git clone test-suit-eavl"
git clone https://github.com/taoyds/test-suite-sql-eval.git

echo "Start evaluation"
python test-suite-sql-eval/evaluation.py --gold dev_gold.txt --pred predicted_sql.txt --etype all --db data/sparc/database --table data/sparc/tables.json