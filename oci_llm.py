#
# see https://python.langchain.com/docs/modules/model_io/models/llms/custom_llm
#
from typing import Any, List, Mapping, Optional
from time import time

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

# import to use OCI GenAI Python API
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import ChatDetails, CohereChatRequest, OnDemandServingMode
from oci.retry import NoneRetryStrategy

class OCIGenAILLM(LLM):
    # added by LS
    model_id: str = "ocid1.generativeaimodel.oc1.eu-frankfurt-1.amaaaaaask7dceyaessaqnexr66avnxjksoibejm2yk2w3rkf7ohicrnx6eq"
    debug: bool = False

    max_tokens: int = 300
    temperature: int = 0
    frequency_penalty: int = 1
    top_p: float = 0.75
    top_k: int = 0
    config: Optional[Any] = None
    service_endpoint: Optional[str] = None
    compartment_id: Optional[str] = None
    timeout: Optional[int] = 10
    signer: Optional[Any] = None

    # moved here by LS
    generative_ai_inference_client: GenerativeAiInferenceClient = None

    """OCI Generative AI LLM model.

    To use, you should have the ``oci`` python package installed, and pass 
    named parameters to the constructor.

    Example:
        .. code-block:: python

            compartment_id = "ocid1.compartment.oc1..."
            CONFIG_PROFILE = "my_custom_profile" # or DEFAULT
            config = oci.config.from_file('~/.oci/config', CONFIG_PROFILE)
            endpoint = "https://generativeai.aiservice.us-chicago-1.oci.oraclecloud.com"
            llm = OCIGenAILLM(
                temperature=0, 
                config=config, 
                compartment_id=compartment_id, 
                service_endpoint=endpoint
                )


    """

    def __init__(self, **kwargs):
        # print(kwargs)
        super().__init__(**kwargs)

        # here we create and store the GenAIClient
        self.generative_ai_inference_client = GenerativeAiInferenceClient(
            config=self.config,
            signer=self.signer,
            service_endpoint=self.service_endpoint,
            retry_strategy=NoneRetryStrategy(),
            timeout=(self.timeout, 240),
        )

    @property
    def _llm_type(self) -> str:
        return "OCI Generative AI LLM"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        tStart = time()

        chat_detail = ChatDetails()
        chat_request = CohereChatRequest()
        chat_request.message = prompt
        chat_request.max_tokens = self.max_tokens
        chat_request.temperature = self.temperature
        chat_request.frequency_penalty = self.frequency_penalty
        chat_request.top_p = self.top_p
        chat_request.top_k = self.top_k

        chat_detail.serving_mode = OnDemandServingMode(model_id=self.model_id)
        chat_detail.chat_request = chat_request
        chat_detail.compartment_id = self.compartment_id

        if self.debug:
            print()
            print("The input prompt is:")
            print(prompt)
            print()

        print("Calling OCI genai (chat)...")
        chat_response = self.generative_ai_inference_client.chat(chat_detail)

        tEla = time() - tStart

        if self.debug:
            print(f"Elapsed time: {round(tEla, 1)} sec...")
            print()

        # TODO: Try to extract the text from the response
        try:
            return chat_response.data.chat_response.text
        except Exception:
            # fallback: print all vars for debugging
            print(vars(chat_response))
            raise

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "frequency_penalty": self.frequency_penalty,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
