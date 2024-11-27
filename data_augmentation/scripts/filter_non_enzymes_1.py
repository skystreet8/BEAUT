from utils import *
from functools import partial
import pandas as pd
from nltk import word_tokenize
import logging


logger = logging.getLogger('filter_non_enzymes_1')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

read_tsv = partial(pd.read_csv, sep='\t', comment='#',
                   names=['query', 'seed_ortholog', 'evalue', 'score','eggNOG_OGs', 'max_annot_lvl', 'COG_category', 'Description',
                          'Preferred_name', 'GOs', 'EC', 'KEGG_ko', 'KEGG_Pathway', 'KEGG_Module', 'KEGG_Reaction', 'KEGG_rclass',
                          'BRITE', 'KEGG_TC', 'CAZy', 'BiGG_Reaction', 'PFAMs'])
KEEP_COG_CATEGORIES = {'C', 'G', 'E', 'F', 'I', 'Q', 'R', 'S'}
filter_ase_words = {'permease', 'peptidase', 'helicase', 'exonuclease', 'atpases', 'heparinase', 'dipeptidyl-peptidase',
                    'dextransucrase', 'sortase', 'endonuclease', 'aminopeptidase', 'oligopeptidase',
                    'oligoendopeptidase', 'carboxypeptidase', 'permeases', 'caspase', 'primase', 'atpase',
                    'pip-5-kinases', "5'-nucleotidase", 'ceramidase', 'metallopeptidase', 'nuclease', 'melibiase',
                    'trehalase', 'amylase', 'xyloglucanase', 'lectin/glucanases', 'proteinase', 'collagenase',
                    'ribonuclease', 'calcium-release', 'terminase', 'excisionase', 'metalloprotease', 'polymerase',
                    'topoisomerase', 'd-transpeptidase', 'depolymerase', 'beta-glucanase', 'elastase',
                    'alpha-trehalase', 'phospholipase_d-nuclease', 'metallocarboxypeptidase', 'gtpase', 'invertase',
                    'endonuclease/exonuclease/phosphatase', 'aminopeptidases', 'acylaminoacyl-peptidases', 'telomerase',
                    'metallo-peptidase', 'replicase', 'fbpase', 'gyrase', 'endo-isopeptidase', 'endoglucanase',
                    '1,4-beta-xylanase', 'hydrogenlyase', 'carnosinase', 'polygalactosaminidase', 'cam-kinase',
                    'chitinase', 'chondroitinase', 'metalloproteinase', 'cyclo-malto-dextrinase', 'gamma-secretase',
                    'metallo-endopeptidase', 'fe-hydrogenase', 'f0f1-atpase', 'trnase', 'l-aminopeptidase', 'dutpase',
                    'inj_translocase', 'ntpase', 'pectinesterase', 'metallo-endoribonuclease', 'exosortase',
                    'beta-1,3-glucanase', 'dgtpase', 'carboxypetidase', 'transcriptase',
                    'protein-phosphocysteine-l-ascorbate-phosphotransferase', 'exoribonuclease', 'coagulase',
                    'recombinase', 'atcase', 'hydrogenase', 'interferase', 'nadase', 'endosialidase', 'nucleases',
                    'transposases', 'dipeptidase', 'nickase/helicase', 'rnase', 'dnase/trnase', '-atpase', 'coprotease',
                    'reactivase', 'metal-chelatase', 'translocase', 'ferrochelatase', 'v-atpase', 'dnase', 'scramblase',
                    'host-nuclease', 'peptidoglycan-synthase', 'chitosanase', 'peptide-n-glycosidase',
                    'cobaltochelatase', 'exodeoxyribonuclease', 'disaggregatase', 'integrase', 'relaxase', 'maturase',
                    'exopeptidases', 'xylanase', 'insulinase', 'metalloendopeptidase', 'arch_atpase', 'de-polymerase',
                    'transposase', 'proteases', 'dapkinase', 'flippase', 'transpeptidase', 'barnase',
                    'polygalacturonase', 'n-atpase', 'ld-carboxypeptidase', 'endopeptidase', 'gyrase/topoisomerase',
                    'autokinase', 'catalase', 'tripeptidases', 'gtpases', 'activase', 'endo-1,4-beta-xylanase',
                    'amylopullulanase', 'aa_permease', 'metallo-carboxypeptidase', 'chelatase', 'aaa-atpase',
                    'protease', 'helicases', 'dna-methyltransferase', 'cellulase', 'endoribonuclease', 'levanase',
                    'excinuclease', 'peptidases', 'convertase', 'apyrase', 'beta-xylanase', 'endodeoxyribonuclease',
                    'dl-endopeptidase', 'lecithinase', 'resolvase', 'primase/polymerase', 'primase-helicase',
                    'cutinase', '1,4-beta-cellobiosidase', 'dipeptidylpeptidase', 'd-aminopeptidase',
                    'amylopullulanase'}

organisms = ['B_Adolescentis', 'B_Xylanisolvens', 'C_Comes', 'C_M62_1', 'H_Filiformis', 'R_Gnavus', 'S_Infantarius']
accessions = ['GCF_000010425.1', 'GCA_000210075.1', 'GCA_000155875.1', 'GCF_000159055.1', 'GCA_000157995.1',
              'GCF_009831375.1', 'GCA_000154985.1']

for organism, accession in zip(organisms, accessions):
    logger.info('--------------------------------')
    logger.info(f'Organism: {organism}')
    logger.info(f'Accession: {accession}')
    logger.info('--------------------------------')
    annotation_df = read_tsv(f'../data/BA_transformers/{accession}.tsv')
    genome_headers, genome_seqs = ReformatFastaFile(f'../data/BA_transformers/{accession}.faa')
    genome_headers = [re.split(r'\s+', h)[0] for h in genome_headers]
    assert len(set(genome_headers) & set(annotation_df['query'].values.tolist())) == len(annotation_df)

    filtered_dfs = []
    logger.info(f'Total annotated sequences: {len(annotation_df)}')
    df_with_ec = annotation_df.query('EC != "-"').copy()
    logger.info(f'Keep {len(df_with_ec)} sequences which have annotated EC numbers.')
    df_wo_ec = annotation_df.query('EC == "-"').copy()
    df_with_cog = df_wo_ec.query('COG_category != "-"').copy()
    logger.info(f'{len(df_with_cog)} sequences have assigned COG categories and no EC numbers.')
    df_wo_cog = df_wo_ec.query('COG_category == "-"').copy()
    logger.info(f'Keep {len(df_wo_cog)} sequences which have no COG categories or EC numbers assigned. ')
    cog_cat_keep_indexes = []
    for t in df_with_cog.itertuples():
        cog = t[7]
        for symbol in cog:
            if symbol in KEEP_COG_CATEGORIES:
                cog_cat_keep_indexes.append(t[0])
                break
    df_keep_by_cog = df_with_cog.loc[cog_cat_keep_indexes].copy()
    logger.info(f'{len(df_keep_by_cog)} sequences passed COG category based filtering.')
    df_remove_by_cog = df_with_cog.drop(cog_cat_keep_indexes)
    ase_indexes = []
    for t in df_keep_by_cog.itertuples():
        desc = t[8]
        words = word_tokenize(desc)
        for w in words:
            if w.endswith('ase') or w.endswith('ases'):
                ase_indexes.append(t[0])
                break
    ase_df = df_keep_by_cog.loc[ase_indexes].copy()
    logger.info(f'{len(ase_df)} sequences have at least one -ase word in their descriptions.')
    keep_ase_indexes = []
    for t in ase_df.itertuples():
        desc = t[8]
        desc = desc.lower()
        words = word_tokenize(desc)
        for w in words:
            if w in filter_ase_words:
                break
        else:
            keep_ase_indexes.append(t[0])
    ase_df = ase_df.loc[keep_ase_indexes].copy()
    logger.info(f'{len(keep_ase_indexes)} passed -ase word based filtering. Keep these sequences.')
    final_df = pd.concat([df_with_ec, ase_df, df_wo_cog], ignore_index=True)
    filtered_dfs.append(final_df)
    logger.info(f'Keep {len(final_df)} annotated sequences in total.')

    merged_df = pd.concat(filtered_dfs, ignore_index=True)
    header2seq = {}
    for h, s in zip(genome_headers, genome_seqs):
        header2seq[h] = s
    seqs = [header2seq[h] for h in merged_df['query'].values.tolist()]
    merged_df = merged_df.assign(seq=seqs)
    merged_df.drop_duplicates(subset=['seq'], inplace=True)

    names = merged_df.columns.values.tolist()
    names = [names[0]] + [names[-1]] + names[1:-1]
    merged_df = merged_df[names]

    unannotated_data = []
    annotated_seqs = set([header2seq[h] for h in annotation_df['query'].values.tolist()])
    for h, s in zip(genome_headers, genome_seqs):
        if not s in annotated_seqs:
            unannotated_data.append([h, s] + ['-' for _ in range(20)])
    unannotated_df = pd.DataFrame(unannotated_data, columns=merged_df.columns)
    unannotated_df.drop_duplicates(subset=['seq'], inplace=True)

    merged_df = pd.concat([merged_df, unannotated_df], ignore_index=True)
    merged_df.fillna('-', inplace=True)
    merged_df.drop_duplicates(subset=['seq'], inplace=True)

    merged_df['length'] = merged_df['seq'].apply(len)
    merged_df = merged_df.query('157 <= length <= 1074').copy()
    del merged_df['length']
    logger.info(f'Number of sequences after filtering and merging unannotated sequences: {len(merged_df)}')
    headers, seqs = merged_df['query'].values.tolist(), merged_df['seq'].values.tolist()
    SaveFastaFile(f'../data/BA_transformers/{organism}_filtered.fasta', headers, seqs)
    merged_df.to_csv(f'../data/BA_transformers/{organism}_filtered_with_annotations.csv', index=False)
