#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

FILES="`find \"$SCRIPT_DIR/../content/\" -name "*.tex" | sort` glossary.tex"

for FILE in $FILES; do
        echo "============="
        echo $FILE
        echo "============="

        # List uses of "an" and "a" in combination with vowels
        ack -i --color-match=on_magenta "($|\\s)(a\\s+[aeiou])" $FILE
        #ack -i --color-match=on_green "($|\\s)(an\\s+[aeiou])" $FILE

        # Uppercase after colon
        ack --color-match=on_blue ":\\s+[A-Z]" $FILE

        # British english
        ack -i --color-match=on_red '(?!size).iz' $FILE

        # Unwanted word repetition
        ack -i --color-match=on_yellow "\b((\w*[aeiou]\w*)(?:\s+\2\b)+)" $FILE

        echo
done


