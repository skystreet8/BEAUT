import re
from utils import *
from functools import partial
import pandas as pd
from nltk import word_tokenize
import logging

logger = logging.getLogger('filter_non_enzymes_2')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

read_tsv = partial(pd.read_csv, sep='\t', comment='#',
                   names=['query', 'seed_ortholog', 'evalue', 'score','eggNOG_OGs', 'max_annot_lvl', 'COG_category', 'Description',
                          'Preferred_name', 'GOs', 'EC', 'KEGG_ko', 'KEGG_Pathway', 'KEGG_Module', 'KEGG_Reaction', 'KEGG_rclass',
                          'BRITE', 'KEGG_TC', 'CAZy', 'BiGG_Reaction', 'PFAMs'])
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

for pt in range(1, 15):
    df = read_tsv(f'../data/PRJNA28331_aug/pt{pt}.emapper.annotations')
    organism_df = pd.read_csv(
        f'../data/PRJNA28331_aug/PRJNA28331_filtered_proteins_positive_results_aug_pt{pt}.csv')
    headers, seqs = ReformatFastaFile(
        f'../data/PRJNA28331_aug/PRJNA28331_filtered_proteins_positive_results_aug_pt{pt}.fasta')
    headers = [re.split(r'\s+', h)[0] for h in headers]
    assert len(set(headers) & set(df['query'].values.tolist())) == len(df)

    logger.info(f'Total annotated sequences: {len(df)}')
    df_with_ec = df.query('EC != "-"').copy()
    logger.info(f'Keep {len(df_with_ec)} sequences which have annotated EC numbers.')
    df_wo_ec = df.query('EC == "-"').copy()
    df_wo_desc = df_wo_ec.query('Description == "-"').copy()
    df_with_desc = df_wo_ec.query('Description != "-"').copy()
    logger.info(f'{len(df_with_desc)} sequences have descriptions and no EC numbers.')
    logger.info(f'Keep {len(df_wo_desc)} sequences which have no descriptions or EC numbers assigned. ')
    keep_ase_indexes = []
    discard_ase_indexes = []
    for t in df_with_desc.itertuples():
        desc = t[8]
        desc = desc.lower()
        words = word_tokenize(desc)
        if not any([w.endswith('ase') or w.endswith('ases') for w in words]):
            # Only keep sequences whose description contains at least one ase-word
            continue
        for w in words:
            if w in filter_ase_words:
                discard_ase_indexes.append(t[0])
                break
        else:
            keep_ase_indexes.append(t[0])
    keep_ase_indexes_ = set(keep_ase_indexes)
    discard_ase_indexes_ = set(discard_ase_indexes)
    remove_indexes = [i for i in df_with_desc.index if i not in keep_ase_indexes_ and i not in discard_ase_indexes_]
    ase_df = df_with_desc.loc[keep_ase_indexes].copy()
    not_ase_df = df_with_desc.loc[remove_indexes].copy()
    logger.info(f'Discard {len(discard_ase_indexes)} sequences whose descriptions contain irrelevant -ase words.')
    logger.info(f'Keep {len(ase_df)} sequences that passed -ase word based filtering.')
    recover_indexes = []
    for t in not_ase_df.itertuples():
        desc = t[8].lower()
        # Recover uncharacterized sequences
        if 'unknown function' in desc or 'DUF' in t[8] or 'uncharacterised' in desc \
            or t[8].startswith('Psort location') or 'non supervised orthologous group' in t[8]:
            recover_indexes.append(t[0])
    logger.info(f'Keep {len(recover_indexes)} sequences whose function are not fully characterised.')
    recovered_df = not_ase_df.loc[recover_indexes].copy()
    final_df = pd.concat([df_with_ec, ase_df, df_wo_desc, recovered_df], ignore_index=True)
    logger.info(f'Keep {len(final_df)} sequences in total.')

    annotated_headers = set(df['query'].values.tolist())
    unannotated_indexes = [t[0] for t in organism_df.itertuples() if t[1] not in annotated_headers]
    logger.info(f'{len(unannotated_indexes)} sequences could not be annotated by EggNOG-mapper.')
    unannotated_df = organism_df.loc[unannotated_indexes].copy()

    final_df.set_index('query', inplace=True)
    organism_df.set_index('header', inplace=True)
    merged_df = organism_df.loc[final_df.index].join(final_df, how='left')
    merged_df.fillna('-', inplace=True)
    merged_df.reset_index(inplace=True)
    merged_df.columns = ['header'] + merged_df.columns.values.tolist()[1:]

    full_df = pd.concat([merged_df, unannotated_df], ignore_index=True)
    full_df.fillna('-', inplace=True)
    logger.info(f'Number of sequences after filtering and merging unannotated sequences: {len(full_df)}')
    full_df.to_csv(
        f'../data/PRJNA28331_aug/PRJNA28331_filtered_proteins_positive_results_aug_filtered_by_annotation_pt{pt}.csv',
        index=False)
    pos_headers, pos_seqs = full_df['header'].values.tolist(), full_df['seq'].values.tolist()
    SaveFastaFile(
        f'../data/PRJNA28331_aug/PRJNA28331_filtered_proteins_positive_results_aug_filtered_by_annotation_pt{pt}.fasta',
        pos_headers, pos_seqs)
