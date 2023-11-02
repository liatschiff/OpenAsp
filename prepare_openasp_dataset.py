import argparse
import gzip
import hashlib
import json
import logging
import os
import random
import re
import sys
import tarfile
import zipfile
from collections import defaultdict
from copy import deepcopy
from typing import Dict
from typing import List, Sequence, Iterable, Set

import requests
import spacy
from sacrerouge.datasets.duc_tac.duc2001 import tasks as duc2001_tasks
from sacrerouge.datasets.duc_tac.duc2002 import metrics as duc2002_metrics
from sacrerouge.datasets.duc_tac.duc2002 import tasks as duc2002_tasks
from sacrerouge.datasets.duc_tac.duc2006 import metrics as duc2006_metrics
from sacrerouge.datasets.duc_tac.duc2006 import task1 as duc2006_tasks
from sacrerouge.datasets.duc_tac.duc2007 import metrics as duc2007_metrics
from sacrerouge.datasets.duc_tac.duc2007 import tasks as duc2007_tasks
from sacrerouge.datasets.multinews import setup as multinews_download_task
from sacrerouge.io import JsonlWriter
from unidecode import unidecode


logger = logging.getLogger()

NLP = None

DUC_ZIP_FILES = (
    'DUC2001_Summarization_Documents.tgz',
    'DUC2002_Summarization_Documents.tgz',
    'DUC2006_Summarization_Documents.tgz',
    'DUC2007_Summarization_Documents.tgz',
)


DOWNLOAD_URLS_2002 = (
    'https://duc.nist.gov/past_duc/duc2002/data/test/summaries/duc2002extractsandabstracts.tar.gz',
    'https://duc.nist.gov/past_duc/duc2002/results/abstracts/phase1/SEEpeers/manualpeers.tar.gz',
)
DOWNLOAD_URLS_2006_2007 = (
    'https://duc.nist.gov/past_duc_aquaint/duc2006/results/NIST/NISTeval.tar.gz',
    'https://duc.nist.gov/past_duc_aquaint/duc2006/results/NIST-secondary-automatic/NISTeval2.tar.gz',
    'https://duc.nist.gov/past_duc_aquaint/duc2006/testdata/duc2006_topics.sgml',
    'https://duc.nist.gov/past_duc_aquaint/duc2007/results/mainEval.tar.gz',
    'https://duc.nist.gov/past_duc_aquaint/duc2007/results/updateEval.tar.gz',
    'https://duc.nist.gov/past_duc_aquaint/duc2007/testdata/duc2007_topics.sgml',
    'https://duc.nist.gov/past_duc_aquaint/duc2007/testdata/duc2007_UPDATEtopics.sgml',
)

DUC_DOCS_INPUT_FNAMES = (
    ('DUC-2001', os.path.join('2001', 'task2.train.200.jsonl')),
    ('DUC-2001', os.path.join('2001', 'task2.test.200.jsonl')),
    ('DUC-2002', os.path.join('2002', 'task2.200.jsonl')),
    ('DUC-2006', os.path.join('2006', 'task1.jsonl')),
    ('DUC-2007', os.path.join('2007', 'task1.jsonl')),
)

DUC_SUMMARIES_INPUT_FNAMES = (
    ('DUC-2001', os.path.join('2001', 'task2.train.200.jsonl')),
    ('DUC-2001', os.path.join('2001', 'task2.test.200.jsonl')),
    ('DUC-2002', os.path.join('2002', 'task2.200.summaries.jsonl')),
    ('DUC-2006', os.path.join('2006', 'task1.summaries.jsonl')),
    ('DUC-2007', os.path.join('2007', 'task1.summaries.jsonl')),
)

MNEWS_INPUT_FNAMES = (
    ('train', 'train.jsonl.gz'),
    ('valid', 'valid.jsonl.gz'),
    ('test', 'test.jsonl.gz'),
)


# the following duc20xx methods are a simplified version of the sacrerouge python code
def duc2002_load_summaries(results_dir):
    mds_abs_summaries = defaultdict(lambda: defaultdict(dict))

    with tarfile.open(os.path.join(results_dir, 'manualpeers.tar.gz'), 'r') as tar:
        for member in tar.getmembers():
            if member.name.endswith('.html'):
                path = member.name.split('/')
                filename = path[-1]
                parts = filename.split('.')

                sentences = duc2002_metrics.load_see_file(tar, member)
                summarizer_id = parts[4]
                if summarizer_id.isalpha():
                    summarizer_type = 'reference'
                else:
                    summarizer_type = 'peer'
                summary = {
                    'summarizer_id': summarizer_id,
                    'summarizer_type': summarizer_type,
                    'text': sentences,
                }

                if parts[1] == 'M':
                    instance_id = parts[0].lower()
                    length = parts[2]
                    if length == '010':
                        length = '10'
                    if length == '050':
                        length = '50'
                    mds_abs_summaries[instance_id][length][summarizer_id] = summary

    return mds_abs_summaries


def duc2002_save_mds_metrics(
    summaries: Dict[str, Dict[str, Dict[str, dict]]], task_name: str, output_dir: str
):
    for length in ['10', '50', '100', '200']:
        with JsonlWriter(
            os.path.join(output_dir, f'{task_name}.{length}.summaries.jsonl')
        ) as out_summaries:
            for instance_id in sorted(summaries.keys()):
                for summarizer_id in summaries[instance_id][length].keys():
                    summary = summaries[instance_id][length][summarizer_id]
                    if len(summary) == 0:
                        continue

                    references = duc2002_metrics.get_mds_references(
                        summaries, instance_id, length, summarizer_id
                    )

                    out_summaries.write(
                        {
                            'instance_id': instance_id,
                            'summarizer_id': summarizer_id,
                            'summarizer_type': summary['summarizer_type'],
                            'summary': summary,
                            'references': references,
                        }
                    )


def duc2006_save_metrics(summaries: Dict[str, Dict[str, dict]], output_dir: str):
    with JsonlWriter(os.path.join(output_dir, 'task1.summaries.jsonl')) as out_summaries:
        for instance_id in sorted(summaries.keys()):
            for summarizer_id in summaries[instance_id].keys():
                summary = summaries[instance_id][summarizer_id]
                references = duc2006_metrics.get_references(summaries, instance_id, summarizer_id)

                out_summaries.write(
                    {
                        'instance_id': instance_id,
                        'summarizer_id': summarizer_id,
                        'summarizer_type': summary['summarizer_type'],
                        'summary': summary,
                        'references': references,
                    }
                )


def duc_2002_extract_and_save_summaries(results_dir, output_dir):
    mds_abs_summaries = duc2002_load_summaries(results_dir)
    duc2002_save_mds_metrics(mds_abs_summaries, 'task2', output_dir)


def duc_2006_extract_and_save_summaries(eval_tar_2, output_dir):
    summaries = duc2006_metrics.load_summaries(eval_tar_2)
    duc2006_save_metrics(summaries, output_dir)


def duc_2007_extract_and_save_summaries(main_eval_tar, output_dir):
    summaries = duc2007_metrics.load_main_summaries(main_eval_tar)
    duc2006_save_metrics(summaries, output_dir)


def download_nist_urls(
    duc_year: str, nist_duc_user: str, nist_duc_password: str, urls: Sequence[str], workdir: str
):
    logger.info(f'downloading {len(urls)} files using {duc_year} credentials')
    for url in urls:
        dst = os.path.join(workdir, 'duc.nist.gov', url.rsplit('/')[-1])
        if os.path.exists(dst):
            logger.info(f'skipping {dst} - already exist')
            continue

        logger.info(f'downloading {url} into {dst}')

        with open(dst, 'wb') as fp:
            resp = requests.get(url, timeout=300, auth=(nist_duc_user, nist_duc_password))
            resp.raise_for_status()
            fp.write(resp.content)
    logger.info('done downloading urls')


def make_workdir(workdir: str):
    logger.info(f'using work dir {workdir}')
    logger.info('creating work dir')
    os.makedirs(workdir, exist_ok=True)

    for year in ['2001', '2002', '2006', '2007']:
        os.makedirs(os.path.join(workdir, 'duc', year), exist_ok=True)
    os.makedirs(os.path.join(workdir, 'mnews'), exist_ok=True)
    os.makedirs(os.path.join(workdir, 'duc.nist.gov'), exist_ok=True)
    return workdir


def extract_duc_zip(nist_duc_zip: str, workdir: str) -> List[str]:
    logger.info('extracting zip file')
    filenames = []
    for file in DUC_ZIP_FILES:
        dst = os.path.join(workdir, file)
        filenames.append(dst)

        if os.path.exists(dst):
            logger.info(f'skipping {file} already exist in {dst}')
            continue

        with zipfile.ZipFile(nist_duc_zip, 'r') as zfile:
            logger.info(f'extracting {file} into {dst}')
            zfile.extract(file, workdir)
    logger.info('successfully extracted zip files')

    return filenames


def extract_duc(workdir: str):
    if all(
        os.path.exists(os.path.join(workdir, 'duc', f))
        for _, f in (DUC_DOCS_INPUT_FNAMES + DUC_SUMMARIES_INPUT_FNAMES)
    ):
        logger.info('skip extracting of duc files - already exist')
        return

    web_results_dir = os.path.join(workdir, 'duc.nist.gov')

    logger.info('extracting duc2001 files')
    duc2001_tasks.main(
        documents_tar=os.path.join(workdir, DUC_ZIP_FILES[0]),
        output_dir=os.path.join(workdir, 'duc', '2001'),
    )

    logger.info('extracting duc2002 files')
    duc2002_outdir = os.path.join(workdir, 'duc', '2002')
    duc2002_tasks.main(
        documents_tar=os.path.join(workdir, DUC_ZIP_FILES[1]),
        extracts_and_abstracts_tar=os.path.join(
            web_results_dir, 'duc2002extractsandabstracts.tar.gz'
        ),
        output_dir=duc2002_outdir,
    )
    duc_2002_extract_and_save_summaries(web_results_dir, duc2002_outdir)

    logger.info('extracting duc2006 files')
    duc2006_outdir = os.path.join(workdir, 'duc', '2006')
    duc2006_tasks.main(
        documents_tar=os.path.join(workdir, DUC_ZIP_FILES[2]),
        eval_tar=os.path.join(web_results_dir, 'NISTeval.tar.gz'),
        topics_file_path=os.path.join(web_results_dir, 'duc2006_topics.sgml'),
        output_dir=duc2006_outdir,
    )
    duc_2006_extract_and_save_summaries(
        os.path.join(web_results_dir, 'NISTeval2.tar.gz'), duc2006_outdir
    )

    logger.info('extracting duc2007 files')
    duc2007_outdir = os.path.join(workdir, 'duc', '2007')
    duc2007_tasks.main(
        documents_tar=os.path.join(workdir, DUC_ZIP_FILES[3]),
        main_eval_tar=os.path.join(web_results_dir, 'mainEval.tar.gz'),
        update_eval_tar=os.path.join(web_results_dir, 'updateEval.tar.gz'),
        main_topics=os.path.join(web_results_dir, 'duc2007_topics.sgml'),
        update_topics=os.path.join(web_results_dir, 'duc2007_UPDATEtopics.sgml'),
        output_dir=duc2007_outdir,
    )
    duc_2007_extract_and_save_summaries(
        os.path.join(web_results_dir, 'mainEval.tar.gz'), duc2007_outdir
    )


def download_file(url: str, fname: str):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()

        with open(fname, 'wb') as f:
            for chunk in r.iter_content(chunk_size=5 * 2**20):
                if chunk:
                    f.write(chunk)


def download_multinews(workdir: str):
    outdir = os.path.join(workdir, 'mnews')
    if all(os.path.exists(os.path.join(outdir, f)) for _, f in MNEWS_INPUT_FNAMES):
        logger.info('skip mnews download - already exist')
        return

    logger.info('downloading multinews files')
    train_src_fname = os.path.join(outdir, 'raw', 'train.src.cleaned')

    # Multinews stored on google drive but because the train set is large
    # we get instead of the file content a "Cannot scan for viruses" response,
    # so we need to grab the link from the html file and download it after.
    if os.path.exists(train_src_fname):
        logger.info('skipping download of mnews train set - already exist')
    else:
        logger.info('getting multinews train file link from google drive')
        redirect_html = requests.get(
            'https://docs.google.com/uc?export=download',
            params={'id': '1wHAWDOwOoQWSj7HYpyJ3Aeud8WhhaJ7P'},
        ).text

        direct_train_link = (
            re.search(r'action="(https://docs.google.com[^"]*)"', redirect_html)
            .group(1)
            .replace('&amp;', '&')
        )

        logger.info(
            f'downloading multinews train file from {direct_train_link} into {train_src_fname}'
        )
        os.makedirs(os.path.dirname(train_src_fname), exist_ok=True)
        download_file(direct_train_link, train_src_fname)

    multinews_download_task.setup(output_dir=outdir, force=False)


def _get_duc_summaries(record):
    if 'summaries' in record:
        summaries = record['summaries']
        if isinstance(summaries[0], list):
            assert len(summaries) == 1
            summaries = summaries[0]
    else:
        summaries = [
            {'summarizer_id': record['summarizer_id'], 'text': record['summary']['text']}
        ] + record['references']

    annotator_id2summary = {}
    for summary in summaries:
        if 'annotator' in summary:
            annotator_id2summary[summary['annotator']] = summary['text']
        else:
            annotator_id2summary[summary['summarizer_id']] = summary['text']

    assert len(annotator_id2summary) == len(summaries)
    return annotator_id2summary


def load_final_duc_data(workdir: str, guids: Set[str]) -> Iterable[dict]:
    logger.info(f'loading duc generic summaries')

    # create mapping of duc_year/topic_id -> annotator_id
    topic_id2annotator_id = {}
    topic_ids_found = {}
    for guid in guids:
        dataset_id, tid, annotator_id = guid.split('/')[-1].rsplit('.')
        if not dataset_id.startswith('DUC-'):
            continue

        topic_id = f'{dataset_id}/{tid}'
        assert topic_id2annotator_id.get(topic_id) in [
            annotator_id,
            None,
        ], f'non unique topic_id {topic_id}'
        topic_id2annotator_id[topic_id] = annotator_id
        topic_ids_found[topic_id] = 0

    topic_id2docs = {}
    for dataset_id, rel_name in DUC_DOCS_INPUT_FNAMES:
        duc_fname = os.path.join(workdir, 'duc', rel_name)
        logger.info(f'loading {dataset_id} data from {duc_fname}')
        with open(duc_fname) as fp:
            for line in fp:
                record = json.loads(line)
                topic_id = f'{dataset_id}/{record["instance_id"]}'
                assert topic_id2docs.get(topic_id) in [record['documents'], None]
                topic_id2docs[topic_id] = [
                    {'filename': d['filename'], 'text': d['text']} for d in record['documents']
                ]

    for dataset_id, rel_name in DUC_SUMMARIES_INPUT_FNAMES:
        duc_fname = os.path.join(workdir, 'duc', rel_name)
        logger.info(f'loading {dataset_id} data from {duc_fname}')
        with open(duc_fname) as fp:
            for line in fp:
                record = json.loads(line)
                topic_id = f'{dataset_id}/{record["instance_id"]}'

                assert topic_id in topic_id2annotator_id, 'all duc topics must be covered'

                annotator_id = topic_id2annotator_id[topic_id]

                annotator2summary = _get_duc_summaries(record)
                if annotator_id not in annotator2summary or topic_ids_found[topic_id] == 1:
                    continue

                summary_text = annotator2summary[annotator_id]
                topic_ids_found[topic_id] += 1
                yield {
                    'topic_id': topic_id,
                    'documents': deepcopy(topic_id2docs[topic_id]),
                    'summary': summary_text,
                }
    assert all(found == 1 for found in topic_ids_found.values()), {
        t: v for t, v in topic_ids_found.items() if v != 0
    }


def get_nlp():
    global NLP
    if NLP is None:
        NLP = spacy.load('en_core_web_sm')
        NLP.max_length = int(1e9)
    return NLP


def norm_mnews_doc(txt: List[str]) -> List[str]:
    txt = '\n'.join(txt)
    nlp = get_nlp()
    lines = [sent.text.strip() for sent in nlp(txt).sents]

    norm_lines = []
    for line in lines:
        line = unidecode(line, errors='preserve')
        line = line.replace('`', "'").replace('\n', ' ')

        if len(line) <= 4 and norm_lines:
            norm_lines[-1] += f' {line}'
            continue
        norm_lines.append(line)

    return norm_lines


def normalize_mnews_summary(txt: List[str]) -> List[str]:
    if isinstance(txt, list):
        txt = '\n'.join(txt)
    nlp = get_nlp()
    txt = txt.strip()
    if txt[0] in '-–':
        txt = txt[1:].strip()

    txt = [sent.text for sent in nlp(txt).sents]
    return [line for line in txt if line.strip()]


def load_final_mnews_data(workdir: str, guids: Set[str]) -> Iterable[dict]:
    # get all topic ids
    topic_ids_found = {}
    for guid in guids:
        dataset_id, tid, _ = guid.split('/')[-1].rsplit('.')
        if not dataset_id.startswith('mnews-'):
            continue
        topic_id = f'{dataset_id}/{tid}'
        topic_ids_found[topic_id] = 0

    for dataset_id, rel_name in MNEWS_INPUT_FNAMES:
        fname = os.path.join(workdir, 'mnews', rel_name)
        with gzip.open(fname, 'rt') as fp:
            for line in fp:
                record = json.loads(line)

                topic_id = f'mnews-{dataset_id}/{record["instance_id"]}'
                if topic_id not in topic_ids_found:
                    continue

                topic_ids_found[topic_id] += 1

                record['documents'].sort(key=lambda d: d['text'])
                docs = [
                    {'filename': f'doc-{fileid}', 'text': norm_mnews_doc(d['text'])}
                    for fileid, d in enumerate(record['documents'], start=1)
                ]
                yield {
                    'topic_id': topic_id,
                    'documents': docs,
                    'summary': normalize_mnews_summary(record['summary']['text']),
                }

    assert all(found == 1 for found in topic_ids_found.values()), 'malformed mnews source files'


def load_openasp_dataset_metadata(openasp_metadata_fname: str):
    logger.info(f'loading aspect dataset from {openasp_metadata_fname}')
    with open(openasp_metadata_fname) as fp:
        aspect_dataset = json.load(fp)
    logger.info(f'found {len(aspect_dataset)} aspect summaries')
    return aspect_dataset


def prepare_aspect_dataset(aspect_dataset, workdir: str):
    guids = {row['guid'] for row in aspect_dataset}

    # generic summaries dataset
    generic_dataset = list(load_final_duc_data(workdir, guids)) + list(
        load_final_mnews_data(workdir, guids)
    )
    topic_id2record = {r['topic_id']: r for r in generic_dataset}
    assert len(topic_id2record) == len(generic_dataset)

    for row in aspect_dataset:
        generic_record = topic_id2record[row['topic_id']]
        generic_summary = generic_record['summary']
        row['reviewer_modified'] = str(row['reviewer_modified'])
        row['review_comment'] = str(row['review_comment'])
        row['summary_text'] = [
            generic_summary[sent_id] for sent_id in row['summary_sentence_indices']
        ]
        row['generic_summary_text'] = generic_summary
        row['documents'] = deepcopy(generic_record['documents'])
        yield row


def save_aspect_dataset(aspects_dataset: List[dict], output_dir):
    logger.info(f'creating output dir {output_dir}')
    os.makedirs(output_dir, exist_ok=True)
    split2fp = {}

    try:
        for row in aspects_dataset:
            split = row.pop('dataset_split')
            if split not in split2fp:
                split_fname = os.path.join(output_dir, f'{split}.jsonl.gz')
                logger.info(f'writing to {split_fname}')
                split2fp[split] = gzip.open(split_fname, 'wt')

            split2fp[split].write(json.dumps(row))
            split2fp[split].write('\n')
    finally:
        for fp in split2fp.values():
            fp.close()
    logger.info(f'finished saving OpenAsp dataset into {output_dir}')


def get_sha256_digest(s: List[str]) -> str:
    return hashlib.sha256('\n'.join(s).encode('utf8')).hexdigest()


def verify_dataset_integrity(aspects_dataset: List[dict], aspect_dataset_indices: List[dict]):
    logger.info('verifying dataset integrity')
    aspect_dataset_indices = sorted(aspect_dataset_indices, key=lambda x: x['guid'])
    aspects_dataset = sorted(aspects_dataset, key=lambda x: x['guid'])

    logger.info('verifying dataset sizes and guids')
    assert len(aspects_dataset) == len(aspect_dataset_indices)
    assert {x['guid'] for x in aspects_dataset} == {x['guid'] for x in aspect_dataset_indices}
    assert len({x['guid'] for x in aspects_dataset}) == len(aspects_dataset)
    assert len({x['guid'] for x in aspect_dataset_indices}) == len(aspect_dataset_indices)

    logger.info('verifying dataset content hashes')
    id2sample_indices = {x['guid']: x for x in aspect_dataset_indices}
    id2sample = {x['guid']: x for x in aspects_dataset}

    mismatches = 0
    for guid, sample_indices in id2sample_indices.items():
        sample = id2sample[guid]
        expected_hash = sample_indices['content_sha256']
        actual_hash = get_sha256_digest(sample['summary_text'])
        if expected_hash != actual_hash:
            mismatches += 1
            logger.warning(f'content hash differ for guid: {guid}')

    if not mismatches:
        logger.info('all summaries content hashes matches')
    else:
        logger.warning(
            f'found {mismatches}/{len(id2sample)} hashes mismatches. '
            f'this can result from different sentence tokenizers or libraries versions'
        )


def run(
    nist_duc_zip: str,
    nist_duc_user_2002: str,
    nist_duc_password_2002: str,
    nist_duc_user_2006: str,
    nist_duc_password_2006: str,
    openasp_metadata_fname: str,
    output_dir: str,
    workdir: str = 'workdir',
):
    logger.info('start creating dataset')
    workdir = make_workdir(workdir)

    # 1. unpack zip from duc email
    extract_duc_zip(nist_duc_zip, workdir)

    # 2. download duc files with password
    download_nist_urls(
        '2002', nist_duc_user_2002, nist_duc_password_2002, DOWNLOAD_URLS_2002, workdir
    )
    download_nist_urls(
        '2006', nist_duc_user_2006, nist_duc_password_2006, DOWNLOAD_URLS_2006_2007, workdir
    )

    # 3. extract duc files
    extract_duc(workdir)

    # 4. download multinews files
    download_multinews(workdir)

    # 5. prepare openasp metadata dataset from all files
    aspect_dataset_indices = load_openasp_dataset_metadata(openasp_metadata_fname)
    aspects_dataset = list(prepare_aspect_dataset(aspect_dataset_indices, workdir))

    # 6. shuffle samples
    rng = random.Random('aspectsum')
    rng.shuffle(aspects_dataset)

    # 7. save aspect dataset to output_file
    save_aspect_dataset(aspects_dataset, output_dir)

    # 8. verify integrity
    verify_dataset_integrity(aspects_dataset, aspect_dataset_indices)

    logger.info(f'script completed successfully, dataset is available on: {output_dir}')


def parse_args(argv=None):
    parser = argparse.ArgumentParser('Prepare and download OpenAsp dataset from metadata files')
    # credentials
    parser.add_argument(
        '--nist-duc2001-user', help='DUC2001-02 username provided by NIST', type=str, required=True
    )
    parser.add_argument(
        '--nist-duc2001-password',
        help='DUC2001-02 password provided by NIST',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--nist-duc2006-user', help='DUC2006-07 username provided by NIST', type=str, required=True
    )
    parser.add_argument(
        '--nist-duc2006-password',
        help='DUC2006-07 password provided by NIST',
        type=str,
        required=True,
    )

    # input
    parser.add_argument(
        '--duc-nist-zip',
        help='fwdrequestingducdata.zip provided by NIST (usually via email)',
        type=str,
        default='fwdrequestingducdata.zip',
    )
    parser.add_argument(
        '--openasp-metadata-fname',
        help='OpenAsp\'s json metadata filename',
        type=str,
        default='openasp_v1_dataset_metadata.json',
    )

    # output
    parser.add_argument(
        '--output-dir', help='OpenAsp\'s final dataset output dir', type=str, default='openasp-v1'
    )

    # other
    parser.add_argument(
        '--workdir',
        help='intermediate raw data and cache files for downloading DUC and MultiNews',
        type=str,
        default='workdir',
    )
    parser.add_argument('--log-level', help='log level name', type=str, default='INFO')

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    logging.basicConfig(
        stream=sys.stdout,
        level=args.log_level,
        format='%(asctime)s [%(levelname)s] %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger.info('logging started')

    run(
        nist_duc_zip=args.duc_nist_zip,
        nist_duc_user_2002=args.nist_duc2001_user,
        nist_duc_password_2002=args.nist_duc2001_password,
        nist_duc_user_2006=args.nist_duc2006_user,
        nist_duc_password_2006=args.nist_duc2006_password,
        openasp_metadata_fname=args.openasp_metadata_fname,
        output_dir=args.output_dir,
        workdir=args.workdir,
    )


if __name__ == '__main__':  # pragma: nocover
    main()
