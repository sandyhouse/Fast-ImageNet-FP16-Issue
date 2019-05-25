lines=`ps aux | grep "./python/bin/python -u train.py" | awk '{print $2}'`
for line in $lines
do
  kill -9 $line
done



