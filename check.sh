if grep -Fqvf en.vec en_words; then
    echo $"There are lines in file1 that don’t occur in file2."
fi
