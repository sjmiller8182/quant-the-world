

# get the zip file
wget http://www.rdatasciencecases.org/Spam/SpamAssassinMessages.zip
mkdir data
mv ./SpamAssassinMessages.zip ./data/
# quietly unzip and remove zip file
unzip -q ./data/SpamAssassinMessages.zip -d ./data
rm -f ./data/SpamAssassinMessages.zip
