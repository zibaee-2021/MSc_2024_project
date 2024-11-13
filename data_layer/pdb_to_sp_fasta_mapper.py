from enum import Enum
import re
import time
import json
import zlib
from xml.etree import ElementTree
from urllib.parse import urlparse, parse_qs, urlencode
import requests
from requests.adapters import HTTPAdapter, Retry
from data_layer import data_handler as dh


class IdMapper:
    """
    These 12 functions were copy-pasted from https://www.uniprot.org/help/id_mapping
    and then convert to (4 public and 8 private) methods in a class.
    """

    def __init__(self):
        self.POLLING_INTERVAL = 3
        self.API_URL = "https://rest.uniprot.org"
        self.retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=self.retries))

    def _check_response(self, response):
        try:
            response.raise_for_status()
        except requests.HTTPError:
            print(response.json())
            raise

    def _get_next_link(self, headers):
        re_next_link = re.compile(r'<(.+)>; rel="next"')
        if "Link" in headers:
            match = re_next_link.match(headers["Link"])
            if match:
                return match.group(1)

    def _get_batch(self, batch_response, file_format, compressed):
        batch_url = self._get_next_link(batch_response.headers)
        while batch_url:
            batch_response = self.session.get(batch_url)
            batch_response.raise_for_status()
            yield self._decode_results(batch_response, file_format, compressed)
            batch_url = self._get_next_link(batch_response.headers)

    def _combine_batches(self, all_results, batch_results, file_format):
        if file_format == "json":
            for key in ("results", "failedIds"):
                if key in batch_results and batch_results[key]:
                    all_results[key] += batch_results[key]
        elif file_format == "tsv":
            return all_results + batch_results[1:]
        else:
            return all_results + batch_results
        return all_results

    def _decode_results(self, response, file_format, compressed):
        if compressed:
            decompressed = zlib.decompress(response.content, 16 + zlib.MAX_WBITS)
            if file_format == "json":
                j = json.loads(decompressed.decode("utf-8"))
                return j
            elif file_format == "tsv":
                return [line for line in decompressed.decode("utf-8").split("\n") if line]
            elif file_format == "xlsx":
                return [decompressed]
            elif file_format == "xml":
                return [decompressed.decode("utf-8")]
            else:
                return decompressed.decode("utf-8")
        elif file_format == "json":
            return response.json()
        elif file_format == "tsv":
            return [line for line in response.text.split("\n") if line]
        elif file_format == "xlsx":
            return [response.content]
        elif file_format == "xml":
            return [response.text]
        return response.text

    def _get_xml_namespace(self, element):
        m = re.match(r"\{(.*)\}", element.tag)
        return m.groups()[0] if m else ""

    def _merge_xml_results(self, xml_results):
        merged_root = ElementTree.fromstring(xml_results[0])
        for result in xml_results[1:]:
            root = ElementTree.fromstring(result)
            for child in root.findall("{http://uniprot.org/uniprot}entry"):
                merged_root.insert(-1, child)
        ElementTree.register_namespace("", self._get_xml_namespace(merged_root[0]))
        return ElementTree.tostring(merged_root, encoding="utf-8", xml_declaration=True)

    def _print_progress_batches(self, batch_index, size, total):
        n_fetched = min((batch_index + 1) * size, total)
        print(f"Fetched: {n_fetched} / {total}")

    def submit_id_mapping(self, from_db, to_db, ids):
        request = requests.post(url=f"{self.API_URL}/idmapping/run",
                                data={"from": from_db, "to": to_db, "ids": ",".join(ids)})
        self._check_response(request)
        return request.json()["jobId"]

    def check_id_mapping_results_ready(self, job_id):
        while True:
            request = self.session.get(f"{self.API_URL}/idmapping/status/{job_id}")
            self._check_response(request)
            j = request.json()
            if "jobStatus" in j:
                if j["jobStatus"] == "RUNNING":
                    print(f"Retrying in {self.POLLING_INTERVAL}s")
                    time.sleep(self.POLLING_INTERVAL)
                else:
                    raise Exception(j["jobStatus"])
            else:
                return bool(j["results"] or j["failedIds"])

    def get_id_mapping_results_link(self, job_id):
        url = f"{self.API_URL}/idmapping/details/{job_id}"
        request = self.session.get(url)
        self._check_response(request)
        return request.json()["redirectURL"]

    def get_id_mapping_results_search(self, url):
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        file_format = query["format"][0] if "format" in query else "json"
        if "size" in query:
            size = int(query["size"][0])
        else:
            size = 500
            query["size"] = size
        compressed = (
            query["compressed"][0].lower() == "true" if "compressed" in query else False
        )
        parsed = parsed._replace(query=urlencode(query, doseq=True))
        url = parsed.geturl()
        request = self.session.get(url)
        self._check_response(request)
        results = self._decode_results(request, file_format, compressed)
        total = int(request.headers["x-total-results"])
        self._print_progress_batches(0, size, total)
        for i, batch in enumerate(self._get_batch(request, file_format, compressed), 1):
            results = self._combine_batches(results, batch, file_format)
            self._print_progress_batches(i, size, total)
        if file_format == "xml":
            return self._merge_xml_results(results)
        return results


class UniprotKey(Enum):
    FROM_DB = 'PDB'
    TO_DB = 'UniProtKB'
    FAILED = 'failedIds'
    RESULTS = 'results'
    FROM = 'from'
    PDBID = 'pdb_id'
    TO = 'to'
    ENTRYTYPE = 'entryType'
    PRIM_ACC = 'primaryAccession'
    SEQ = 'sequence'
    VALUE = 'value'
    LEN = 'length'

    SP = 'Swiss-Prot'
    TR = 'TrEMBL'
    SP_ACC_ID = 'sp_acc_id'
    FASTA = 'fasta'


def call_uniprot_api_for_acc_ids_and_sp_fasta(pdb_ids):
    swissprot_recs = dict()

    idmpr = IdMapper()
    jobid = idmpr.submit_id_mapping(from_db=UniprotKey.FROM_DB.value, to_db=UniprotKey.TO_DB.value, ids=pdb_ids)

    if idmpr.check_id_mapping_results_ready(jobid):

        link = idmpr.get_id_mapping_results_link(jobid)
        results_: dict = idmpr.get_id_mapping_results_search(link)

        failed_pdb_ids = results_[UniprotKey.FAILED.value] if UniprotKey.FAILED.value in results_ else None
        if failed_pdb_ids:
            print(f'{len(failed_pdb_ids)} failed: {failed_pdb_ids}')

        results = results_[UniprotKey.RESULTS.value]
        print(f'Number of results = {len(results)}')

        swissprot_count, trembl_count = 0, 0

        for result in results:
            swissprot_rec = dict()
            pdb_id = result[UniprotKey.FROM.value]

            if failed_pdb_ids and pdb_id in failed_pdb_ids:
                swissprot_recs[pdb_id] = None

            else:
                to = result[UniprotKey.TO.value]
                sp_or_tr = to[UniprotKey.ENTRYTYPE.value]

                if UniprotKey.SP.value.lower() in sp_or_tr.lower():
                    swissprot_count += 1
                    swissprot_rec[UniprotKey.SP_ACC_ID.value] = to[UniprotKey.PRIM_ACC.value]
                    fasta_seq = to[UniprotKey.SEQ.value]
                    swissprot_rec[UniprotKey.FASTA.value] = fasta_seq[UniprotKey.VALUE.value]
                    swissprot_rec[UniprotKey.LEN.value] = fasta_seq[UniprotKey.LEN.value]

                    swissprot_recs[pdb_id] = swissprot_rec

                elif UniprotKey.TR.value.lower() in sp_or_tr.lower():
                    swissprot_recs[pdb_id] = None
                    trembl_count += 1
                    # print(f"'Swiss-prot' is not in this `entryType`. Instead `entryType` is: '{sp_or_tr}'.")

        print(f'Number of TrEMBL recs = {trembl_count}.')  # (Expecting 250 x TrEMBL from the 573 sd proteins cifs).
        print(f'Number of Swiss-Prot recs = {swissprot_count}.')  # (Expecting 363 x Swiss-Prot from the 573).
        return swissprot_recs


class Filename(Enum):
    sd_5_globins_txt = 'SD_5_globins.txt'
    pdbid_fasta_5globs = 'PDBid_sp_FASTA_5_Globins'


if __name__ == '__main__':

    # pdbids_single_domain = dh.get_list_of_pdbids_of_locally_downloaded_SD_cifs()
    # swissprot_records = call_uniprot_api_for_acc_ids_and_sp_fasta(pdbids_single_domain)
    # dh.write_pdb_uniprot_fasta_recs_to_json(swissprot_records, filename='pdbids_sp_fastas')

    # READ LIST OF PDB IDS OF GLOBINS FROM TEXT FILE:
    pdbids_sd_5_globins = dh.read_list_of_pdbids_from_text_file(filename=Filename.sd_5_globins_txt.value)
    swissprot_globins = call_uniprot_api_for_acc_ids_and_sp_fasta(pdbids_sd_5_globins)
    dh.write_pdb_uniprot_fasta_recs_to_json(swissprot_globins, filename=Filename.pdbid_fasta_5globs.value)
