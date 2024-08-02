#!/usr/bin/env python3

#  Copyright Red Hat
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

import re
from textwrap import dedent
from typing import Any, Dict

import requests
from langchain_core.messages import BaseMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from .base import ModelMeshClient
from .exceptions import ModelTimeoutError

from rulebook_eval.eval import QualityValidator
from rulebook_eval.postprocess import PostProcessor
from rulebook_eval.object_loader import RulebookLoader
import os
import json

# SYSTEM_MESSAGE_TEMPLATE = (
#     "You are an Ansible expert. Return a single task that best completes the "
#     "partial playbook. Return only the task as YAML. Do not return multiple tasks. "
#     "Do not explain your response. Do not include the prompt in your response."
# )
SYSTEM_MESSAGE_TEMPLATE = ""
HUMAN_MESSAGE_TEMPLATE = "{prompt}"


ENABLE_ADVANCE_EVAL = os.getenv("ENABLE_ADVANCE_EVAL", "False").lower() == "true"
ADV_EVAL_API = os.getenv("ADV_EVAL_API")

qv = QualityValidator()
pp = PostProcessor()
rbl = RulebookLoader()


def unwrap_playbook_answer(message: str | BaseMessage) -> tuple[str, str]:
    task: str = ""
    if isinstance(message, BaseMessage):
        if (
            isinstance(message.content, list)
            and len(message.content)
            and isinstance(message.content[0], str)
        ):
            task = message.content[0]
        elif isinstance(message.content, str):
            task = message.content
    elif isinstance(message, str):
        # Ollama currently answers with just a string
        task = message
    if not task:
        raise ValueError

    m = re.search(r".*?```(yaml|)\n+(.+)```(.*)", task, re.MULTILINE | re.DOTALL)
    if m:
        playbook = m.group(2).strip()
        outline = m.group(3).lstrip().strip()
        return playbook, outline
    else:
        return "", ""


def unwrap_task_answer(message: str | BaseMessage) -> str:
    task: str = ""
    if isinstance(message, BaseMessage):
        if (
            isinstance(message.content, list)
            and len(message.content)
            and isinstance(message.content[0], str)
        ):
            task = message.content[0]
        elif isinstance(message.content, str):
            task = message.content
    elif isinstance(message, str):
        # Ollama currently answers with just a string
        task = message
    if not task:
        raise ValueError

    m = re.search(r"```(yaml|)\n+(.+)```", task, re.MULTILINE | re.DOTALL)
    if m:
        task = m.group(2)
    return dedent(re.split(r"- name: .+\n", task)[-1]).rstrip()


class LangChainClient(ModelMeshClient):
    def get_chat_model(self, model_id):
        raise NotImplementedError

    def infer(self, request, model_input, model_id="", suggestion_id=None) -> Dict[str, Any]:
        model_id = self.get_model_id(request.user, None, model_id)

        prompt = model_input.get("instances", [{}])[0].get("prompt", "")
        context = model_input.get("instances", [{}])[0].get("context", "")

        # NOTE: The change below is just for rulebook PoC, need to update or remove it later
        full_prompt = f"Question:\n{prompt}\nAnswer:\n"
        # full_prompt = f"{context}{prompt}\n"
        
        llm = self.get_chat_model(model_id)

        chat_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    SYSTEM_MESSAGE_TEMPLATE, additional_kwargs={"role": "system"}
                ),
                HumanMessagePromptTemplate.from_template(
                    HUMAN_MESSAGE_TEMPLATE, additional_kwargs={"role": "user"}
                ),
            ]
        )

        try:
            chain = chat_template | llm
            message = chain.invoke({"prompt": full_prompt})
            # response = {"predictions": [unwrap_task_answer(message)], "model_id": model_id}
            response = {"predictions": [message], "model_id": model_id}

            return response

        except requests.exceptions.Timeout:
            raise ModelTimeoutError

    def generate_playbook(
        self,
        request,
        text: str = "",
        create_outline: bool = False,
        outline: str = "",
        generation_id: str = "",
    ) -> tuple[str, str]:
        SYSTEM_MESSAGE_TEMPLATE = """
        You are an Ansible expert.
        Your role is to help Ansible developers write rulebooks.
        You answer with an Ansible rulebook.
        """

        SYSTEM_MESSAGE_TEMPLATE_WITH_OUTLINE = """
        You are an Ansible expert.
        Your role is to help Ansible developers write rulebooks.
        The first part of the answer is an Ansible rulebook.
        the second part is a step by step explanation of this.
        Use a new line to explain each step.
        """

        HUMAN_MESSAGE_TEMPLATE = """
        This is what the rulebook should do: {text}
        """

        HUMAN_MESSAGE_TEMPLATE_WITH_OUTLINE = """
        This is what the rulebook should do: {text}
        This is a break down of the expected Rulebook: {outline}
        """

        system_template = (
            SYSTEM_MESSAGE_TEMPLATE_WITH_OUTLINE if create_outline else SYSTEM_MESSAGE_TEMPLATE
        )
        human_template = HUMAN_MESSAGE_TEMPLATE_WITH_OUTLINE if outline else HUMAN_MESSAGE_TEMPLATE
        from ansible_ai_connect.ai.api.model_client.langchain import (
            unwrap_playbook_answer,
        )

        model_id = self.get_model_id(request.user, None, "")
        llm = self.get_chat_model(model_id)

        chat_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    dedent(system_template),
                    additional_kwargs={"role": "system"},
                ),
                HumanMessagePromptTemplate.from_template(
                    dedent(human_template), additional_kwargs={"role": "user"}
                ),
            ]
        )

        # NOTE: for rulebook PoC
        chain = chat_template | llm
        org_prompt = text
        text = f"Question:\n{text}\nAnswer:\n```yaml"
        print(f"[DEBUG] right before generate_playbook() invoke() text: {text}, outline: {outline}")
        output = chain.invoke({"text": text, "outline": outline})
        print(f"[DEBUG] right after generate_playbook() invoke() output: {output}")
        playbook = output
        outline = ""
        # playbook, outline = unwrap_playbook_answer(output)
        print(f"[DEBUG] right after unwrap_playbook_answer() playbook: {playbook}")
        if not create_outline:
            outline = ""
        
        # Postprocess and Eval
        playbook, pp_detail = post_process(prediction=playbook)
        print(f"[DEBUG] post-processed yaml  output: {playbook}")
        print("[DEBUG] post-process detail:", pp_detail)
        use_adv_eval = True if ENABLE_ADVANCE_EVAL and ADV_EVAL_API else False
        eval_res = evaluation(prompt=org_prompt, prediction=playbook, use_adv_eval=use_adv_eval)   
        print("[DEBUG] evaluation:", eval_res)     
        playbook = add_comment(eval_res=eval_res, prediction=playbook)
        print(f"[DEBUG] commented yaml  output: {playbook}")

        return playbook, outline

    def explain_playbook(self, request, content, explanation_id: str = "") -> str:
        SYSTEM_MESSAGE_TEMPLATE = """
        You're an Ansible expert.
        You format your output with Markdown.
        You only answer with text paragraphs.
        Write one paragraph per Ansible task.
        Markdown title starts with the '#' character.
        Write a title before every paragraph.
        Do not return any YAML or Ansible in the output.
        Give a lot of details regarding the parameters of each Ansible plugin.
        """

        HUMAN_MESSAGE_TEMPLATE = """Please explain the following Ansible playbook:

        {playbook}"
        """

        model_id = self.get_model_id(request.user, None, "")
        llm = self.get_chat_model(model_id)

        chat_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    dedent(SYSTEM_MESSAGE_TEMPLATE),
                    additional_kwargs={"role": "system"},
                ),
                HumanMessagePromptTemplate.from_template(
                    dedent(HUMAN_MESSAGE_TEMPLATE), additional_kwargs={"role": "user"}
                ),
            ]
        )

        chain = chat_template | llm
        explanation = chain.invoke({"playbook": content})
        return explanation


def post_process(prediction):
    separators = ["\n\n\n", "\nASSISTANT:", "###\n", "```"]
    for separator in separators:
        if separator in prediction:
            prediction = prediction.split(separator)[0].strip()
    try:
        rulesets_obj = rbl.load_rulesets_from_yaml(prediction)
        pp_yamls, pp_detail = pp.postprocess_rulesets(rulesets_obj)
        return pp_yamls[0], pp_detail
    except Exception as exc:
            print("[WARNING] postprocess error", exc)
    return prediction, []


def evaluation(prompt, prediction, use_adv_eval=False):
    eval_res = {}
    if use_adv_eval:
        try:
            print("[DEBUG]use advance evaluation")
            payload = {"prompt": prompt, "prediction": prediction}
            response = requests.post(ADV_EVAL_API, data=json.dumps(payload), headers={'Content-Type': 'application/json'})
            if response.status_code == 200:
                print("[DEBUG] Advance eval result:", response.json(), type(response.json()))
                eval_res = response.json()
            else:
                print(f"[WARNING]failed api call to advance eval server {ADV_EVAL_API}", response.status_code)
        except Exception as exc:
            exception = exc
            print(f"[WARNING]failed api call to advance eval server {ADV_EVAL_API}", exception)
    if not use_adv_eval or not eval_res:
        eval_res = qv.evaluate(prediction)
    return eval_res


def add_comment(eval_res, prediction):
    confidence_score = eval_res.get("score")
    syntax_check_conditions = [
        eval_res.get("yaml_parse_ok"),
        eval_res.get("parse_ok"),
        not eval_res.get("non_exist_action"),
        not eval_res.get("wrong_action_args"),
        not eval_res.get("non_exist_sources"),
        not eval_res.get("wrong_source_args")
    ]
    syntax_check = all(syntax_check_conditions)
    comment = f"confidence score: {confidence_score}, syntax check: {syntax_check}"
    try:
        rulesets_obj = rbl.load_rulesets_from_yaml(prediction)
        rulesets_obj[0].comment = comment
        prediction = rulesets_obj[0].to_yaml_with_comment()
    except Exception as exc:
        print("[WARNING] load post-processed prediction as rulebook object", exc)
        lines = prediction.split('\n')
        for i, line in enumerate(lines):
            if 'name:' in line:
                lines.insert(i, f"# {comment}")
                break
        prediction = "\n".join(lines)
    return prediction