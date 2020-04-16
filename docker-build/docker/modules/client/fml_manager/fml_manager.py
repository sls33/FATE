#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#
import json
import os
import tarfile
import requests

from contextlib import closing
from arch.api.utils import file_utils
from arch.api.utils.core import get_lan_ip


class FMLManager:
    def __init__(self, server_conf="/data/projects/fate/python/arch/conf/server_conf.json", log_path="./"):
        self.server_conf = file_utils.load_json_conf(server_conf)
        self.ip = self.server_conf.get("servers").get("fateflow").get("host")
        if self.ip in ['localhost', '127.0.0.1', 'python']:
            self.ip = get_lan_ip()

        self.http_port = self.server_conf.get("servers").get("fateflow").get("http.port")
        self.server_url = "http://{}:{}/{}".format(self.ip, self.http_port, "v1")
        self.log_path = log_path

    def submit_job(self, dsl_path, config_path):
        if config_path:
            config_data = {}
            config_path = os.path.abspath(config_path)
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        else:
            raise Exception('Conf cannot be null.')
        dsl_data = {}
        if dsl_path:
            dsl_path = os.path.abspath(dsl_path)
            with open(dsl_path, 'r') as f:
                dsl_data = json.load(f)
        else:
            raise Exception('DSL_path cannot be null.')

        post_data = {'job_dsl': dsl_data,
                     'job_runtime_conf': config_data}
        response = requests.post("/".join([self.server_url, "job", "submit"]), json=post_data)

        return self.prettify(response)

    def query_job(self, query_conditions):
        response = requests.post("/".join([self.server_url, "job", "query"]), json=query_conditions)
        return self.prettify(response)

    def stop_job(self, job_id):
        post_data = {
            'job_id': job_id
        }
        response = requests.post("/".join([self.server_url, "job", "stop"]), json=post_data)
        return self.prettify(response)

    def fetch_job_log(self, job_id):
        data = {
            "job_id": job_id
        }

        tar_file_name = 'job_{}_log.tar.gz'.format(job_id)
        extract_dir = os.path.join(self.log_path, 'job_{}_log'.format(job_id))
        with closing(requests.get("/".join([self.server_url, "job", "log"]), json=data,
                                      stream=True)) as response:
            if response.status_code == 200:
                self.__download_from_request(http_response=response, tar_file_name=tar_file_name, extract_dir=extract_dir)
                response = {'retcode': 0,
                            'directory': extract_dir,
                            'retmsg': 'download successfully, please check {} directory, file name is {}'.format(extract_dir, tar_file_name)}

                return self.prettify(response, True)
            else:
                return self.prettify(response, True)

    def load_data(self, url, namespace, table_name, work_mode, head, partition):
        post_data = {
            "file": url,
            "namespace": namespace,
            "table_name": table_name,
            "work_mode": work_mode,
            "head": head,
            "partition": partition
        }

        response = requests.post("/".join([self.server_url, "data", "upload"]), json=post_data)

        return self.prettify(response)

    def query_data(self, job_id, limit):
        post_data = {
            "job_id": job_id,
            "limit": limit
        }

        response = requests.post("/".join([self.server_url, "data", "upload", "history"]), json=post_data)

        return self.prettify(response)

    def download_data(self, namespace, table_name, output_path, work_mode, delimitor):
        post_data = {
            "namespace": namespace,
            "table_name": table_name,
            "work_mode": work_mode,
            "delimitor": delimitor,
            "output_path": output_path
        }

        response = requests.post("/".join([self.server_url, "data", "download"]), json=post_data)

        return self.prettify(response)

    def export_data(self):
        print("TBD")

    def prettify(self, response, verbose=False):
        if verbose:
            if isinstance(response, requests.Response):
                if response.status_code == 200:
                    print("Success!")
                print(json.dumps(response.json(), indent=4, ensure_ascii=False))
            else:
                print(response)

        return response

    def __download_from_request(self, http_response, tar_file_name, extract_dir):
        with open(tar_file_name, 'wb') as fw:
            for chunk in http_response.iter_content(1024):
                if chunk:
                    fw.write(chunk)
        tar = tarfile.open(tar_file_name, "r:gz")
        file_names = tar.getnames()
        for file_name in file_names:
            tar.extract(file_name, extract_dir)
        tar.close()
        os.remove(tar_file_name)