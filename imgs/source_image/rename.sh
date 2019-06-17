for i in $(find . -name "*.png")
do
    echo $i
    #echo $i | sed 's/x[248]//'
    mv "$i" "`echo $i | sed 's/x[248]//'`"
done
