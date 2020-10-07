"""Open the gzipped file and write it with the same name
"""

import gzip
import shutil

with gzip.open('./case_8.csv.gz', 'rb') as f_in:
    with open('./case_8.csv', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
