import glob
import csv

##caminho = raw_input("Caminho dos arquivos: ")
##'resultados_turn\Variance\*.txt'

list_caminho =[
    'resultados_flop\Variance',
    'resultados_flop\Percentil',
    'resultados_flop\Select k best',
    'resultados_flop\SELECTFDR',
    'resultados_flop\SELECTFPR',
    'resultados_flop\SELECTFWE'
]
def edita_feature(feature):
    feature = feature.replace("'", "")
    feature = feature.replace("[", "")
    feature = feature.replace("]", "")

    return feature

def salva_result(caminho, rank, media_total):
    fileWrite = open(caminho+'\_result.txt', "a")
    fileWrite.write("Media de atributos selecionados: "+str(media_total)+"\n Atributos: "+str(rank[:media_total]))
    fileWrite.close()

for caminho in list_caminho:
    arqs = glob.glob(caminho+'\*.txt')

    rank = {}
    media_total = 0
    for arq in arqs:
        print(arq)
        dados = []
        with open(arq) as csvfile:
            array = csv.reader(csvfile, delimiter=' ')

            for l in array:
                dados.append(l)


        dados = dados[1:]
        media = 0
        for features in dados:
            edita_feature(features[-1])
            media += float(features[-1])
            features = features[:-1]
            for i in range(len(features)):
                features[i] = edita_feature(features[i])
                if features[i] in rank:
                    rank[features[i]] += 1
                else:
                    rank[features[i]] = 1

        media_total += (media/10)

    rank = sorted(rank.items(), key = lambda e: (-e[1], e[0]))
    salva_result(caminho,rank, int(round(media_total/5)))




