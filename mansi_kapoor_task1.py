from pyspark import SparkContext
import sys
import itertools
from time import time

input_file_path = sys.argv[1]
output_file_path = sys.argv[2]

def process(entry):
    revisedEntries= entry[0].split(',')
    return (revisedEntries[0], revisedEntries[1], revisedEntries[2])

def generate_minhash_Array(users,num_hashes, num_users):
    users = list(users)
    system_max_value = sys.maxsize
    hashed_users = [system_max_value for i in range(0,num_hashes)]

    for user in users:
        for i in range(1, num_hashes+1):
            current_hash_code = ((i*user)+ (5*i*13)) % num_users

            if current_hash_code < hashed_users[i-1]:
                hashed_users[i-1] = current_hash_code
    return hashed_users

def applyLSHToSignature(business_id, signatures, n_bands, n_rows):
    signature_tuples = []
    for band in range(0,n_bands):
        band_name = band
        final_signature = signatures[band*n_rows:(band*n_rows)+n_rows]
        final_signature.insert(0, band_name)
        signature_tuple = (tuple(final_signature), business_id)
        signature_tuples.append(signature_tuple)

    return signature_tuples

def generate_similar_businesses(businesses):

    similar_businesses = sorted(list(itertools.combinations(sorted(businesses),2)))

    return similar_businesses


def calculate_jaccard(candidate, business_char_matrix):

    users_c1 = set(business_char_matrix.get(candidate[0]))
    users_c2 = set(business_char_matrix.get(candidate[1]))

    jaccard_intersection = len(users_c1.intersection(users_c2))
    jaccard_union = len(users_c1.union(users_c2))

    jaccard_similarity_value = float(jaccard_intersection)/float(jaccard_union)

    return (candidate, jaccard_similarity_value)

result = []

sc = SparkContext('local[*]', 'task1')


jaccard_support = 0.5
num_hashes = 80
n_bands = 40
n_rows = 2

start = time()
user_businessRdd = sc.textFile(input_file_path).map(lambda entry: entry.split('\n')).map(lambda entry: process(entry))
headers = user_businessRdd.first()
finalRdd = user_businessRdd.filter(lambda entry: entry != headers)

users = finalRdd.map(lambda entry: entry[0]).distinct()
num_users = users.count()

user_index_dict = finalRdd.map(lambda entry: entry[0]).zipWithIndex().collectAsMap()


business_user_map = finalRdd\
    .map(lambda entry: (entry[1], user_index_dict.get(entry[0])))\
    .groupByKey()\
    .sortByKey()\
    .mapValues(lambda entry: set(entry))\
    .persist()

business_user_map_values = business_user_map.collect()
business_char_matrix = {}

for bu in business_user_map_values:
    business_char_matrix.update({bu[0]:bu[1]})


candidates = business_user_map\
    .mapValues(lambda users: generate_minhash_Array(users, num_hashes, num_users))\
    .flatMap(lambda entry: applyLSHToSignature(entry[0], list(entry[1]), n_bands, n_rows))\
    .groupByKey()\
    .filter(lambda entry: len(list(entry[1])) > 1)\
    .flatMap(lambda entry: generate_similar_businesses(sorted(list(entry[1]))))\
    .distinct()\
    .persist()

final_pairs = candidates\
    .map(lambda cd: calculate_jaccard(cd, business_char_matrix))\
    .filter(lambda cd: cd[1] >= jaccard_support)\
    .sortByKey()

result = final_pairs.collect()


with open(output_file_path, "w") as fp:
    fp.write("business_id_1,business_id_2,similarity")
    fp.write('\n')
    fp.write('\n'.join('{},{},{}'.format(x[0][0], x[0][1], x[1]) for x in result))

end = time()
print("Count: ", len(result))
print("Duration: " + str(end-start))