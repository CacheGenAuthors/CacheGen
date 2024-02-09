from typing import Tuple, TypeAlias, Union, List
from dataclasses import dataclass
import pickle
import hashlib
import torch
import torchac
import socket
from transformers import AutoTokenizer, AutoModelForCausalLM
KVCache: TypeAlias = Tuple[Tuple[torch.Tensor]]
import time
import subprocess
import os


def vectorwise_quant(xq, max1, dim=-1, quant_type="vector"):
    
    C = int(os.environ["BINS"]) // 2 - 1

    x = (xq / C * max1).to(torch.float16)
    return x

@dataclass
class CacheGenConfig:
    """
    The configuration of how to do prefill in cache gen
    Fields:
    - start_index, end_index: indicating the data (either kv cache or tokenized input) corresponds to 
                              [start_index, end_index) of the whole tokenized input (i.e., input_ids[start_index:end_index])
    - is_kv: a boolean value, indicating if the underlying data is kv cache or tokenized input (input ids)
    - data: can be either kv cache (usually is a Tuple[Tuple[torch.Tensor, torch.Tensor]]) or tokenized input (just torch.Tensor)

    NOTE: we do not support splitting on raw text data, because the mapping between text and tokenized text is not deterministic.
          For example, "我吃水果" might be tokenized to 3 tokens where "水果" is mapped to a single token. But if
          we split the raw text to ["我", "吃", "水", "果"], it will be mapped to 4 tokens.
    """

    ''' start index on the whole tokenized input, inclusive '''
    start_index: int

    ''' end index on the whole tokenized input, exclusive '''
    end_index: int

    ''' the flag indicating whether the data is kv cache or tokenized input '''
    is_kv: bool



class CacheGenEngine:
    """
    The CacheGen engine
    - check if text/token list exists in cache
    - given text/token list, retrieve kv cache chunks
    - given text/token list and kv, store the kv into cachegen engine
    """

    def __init__(self, *args, **kwargs):
        # TODO: initialize the cache gen engine
        # pass
        self.cdf_k = pickle.load(open("bits/normalized_cdf_layer0_key.pkl", "rb"))
        self.cdf_v = pickle.load(open("bits/normalized_cdf_layer0_value.pkl", "rb"))
        self.cdf_k2 = pickle.load(open("bits/normalized_cdf_layer1_key.pkl", "rb"))
        self.cdf_v2 = pickle.load(open("bits/normalized_cdf_layer1_value.pkl", "rb"))
        self.cdf_k3 = pickle.load(open("bits/normalized_cdf_layer2_key.pkl", "rb"))
        max_tensors1 = pickle.load(open("bits/max_layer0_key.pkl", "rb"))
        max_tensors2 = pickle.load(open("bits/max_layer0_value.pkl", "rb"))
        tmp = [ ]
        for i in max_tensors1:
            tmp.append(max_tensors1[i].unsqueeze(0))
        self.max_k1 = torch.cat(tmp, dim=0).cuda()
        tmp = [ ]
        for i in max_tensors2:
            tmp.append(max_tensors2[i].unsqueeze(0))
        self.max_v1 = torch.cat(tmp, dim=0).cuda()
        
        
        self.model = AutoModelForCausalLM.from_pretrained(kwargs["model_name"], load_in_8bit= True)
        self.model.eval()
        # self.model.half()
        # self.model.cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(kwargs["model_name"])
    def transformer_kv_to_tuple(self, key, value):
        kv_list = []
        for i in range(len(key)):
            tmp1 = key[i].reshape((key[i].shape[0], 32, 128)).permute((1, 0, 2)).unsqueeze(0)
            tmp2 = value[i].reshape((key[i].shape[0], 32, 128)).permute((1, 0, 2)).unsqueeze(0)
            kv_list += [(tmp1, tmp2)]
        return tuple(kv_list)
    def input_ids_to_key(self, input_ids: torch.Tensor) -> str:
        """
        Compute the unique key based on the value of the input_ids
        """
        # TODO: this is a helper function, please implement your own logic here
        tensor_bytes = input_ids.cpu().numpy().tobytes()
        return hashlib.sha256(tensor_bytes).hexdigest()
    
    def contains(self, input_ids: torch.Tensor) -> bool:
        """
        Check if the input ids is in the cache gen engine

        Input:
            input_ids: a torch.Tensor whose shape of N elements. We assume that there is NO batching dimension

        Returns:
            True if the input ids is in the cache gen engine, False otherwise
        """
        # TODO: maybe do the "prefix check" here
        pass
    def decode_from_pt(self, input_path, input_path2, input_path3, max1):
        st = time.monotonic()
        decoded_kv1 = torch.load(input_path).half()
        decoded_kv2 = torch.load(input_path2).half()
        decoded_kv3 = torch.load(input_path3).half()
        end = time.monotonic()
        os.environ['BINS'] = "64"
        for l in range(decoded_kv1.shape[1]):
            decoded_kv1[:, l] = vectorwise_quant(decoded_kv1[:, l], max1[l].cuda()) 
        decoded_kv1 = decoded_kv1.permute(1, 0, 2)
        
        
        
        os.environ['BINS'] = "32"
        for l in range(decoded_kv2.shape[1]):
            decoded_kv2[:, l] = vectorwise_quant(decoded_kv2[:, l], max1[l + 10].cuda()) 
        decoded_kv2 = decoded_kv2.permute(1, 0, 2)
        
        
        os.environ['BINS'] = "16"
        for l in range(decoded_kv3.shape[1]):
            decoded_kv3[:, l] = vectorwise_quant(decoded_kv3[:, l], max1[l + 20].cuda()) 
        decoded_kv3 = decoded_kv3.permute(1, 0, 2)
        return decoded_kv1, decoded_kv2, decoded_kv3, end-st
        return decoded_kv1, decoded_kv2, decoded_kv3
    def decode(self,  cdf, cdf2, cdf3, input_bits, input_bits2, input_bits3, decoded_kv1, decoded_kv2, decoded_kv3) -> KVCache:
        st = time.monotonic()
        torchac.test(cdf, input_bits, 1000, 10, 100)
        torchac.test(cdf2, input_bits2, 1000, 10, 100)
        torchac.test(cdf3, input_bits3, 1000, 10, 100)
        key_cache, value_cache = self.transform_tuple_to_kv(self.past_kv)
        # key_cache[:10] = decoded_kv1
        # key_cache[10:20] = decoded_kv2
        # key_cache[20:] = decoded_kv3
        kv_cache = self.transformer_kv_to_tuple(key_cache, value_cache)
        return kv_cache
    
    # def get(self, input_ids: torch.Tensor) -> List[CacheGenConfig]:
    #     """
    #     Get the KV cache from the input ids
    #     Will be called by CacheGenController.get()
    #     """
    #     # TODO: maybe do the decompression here? Also you can mix the kv cache and text

        
    #     prefill_kv = None
    #     import numpy as np
    #     st = time.monotonic()
    #     max1 = pickle.load(open("max.pkl", 'rb'))
    #     bitstreams = pickle.load(open("L3C-PyTorch/src/bitstreams_layer1.pkl", "rb"))
    #     cdf = pickle.load(open("L3C-PyTorch/src/normalized_cdf_layer1.pkl", "rb"))
    #     bitstreams2 = pickle.load(open("L3C-PyTorch/src/bitstreams_layer2.pkl", "rb"))
    #     cdf2 = pickle.load(open("L3C-PyTorch/src/normalized_cdf_layer2.pkl", "rb"))
    #     bitstreams3 = pickle.load(open("L3C-PyTorch/src/bitstreams_layer3.pkl", "rb"))
    #     cdf3 = pickle.load(open("L3C-PyTorch/src/normalized_cdf_layer3.pkl", "rb"))
    #     input_bits, input_bits2, input_bits3 = [], [], []
    #     for i in range(1000):
    #         input_bits.append(bitstreams[i])
    #         input_bits2.append(bitstreams2[i])
    #         input_bits3.append(bitstreams3[i])
    #     max1 = [max1[x].cuda().unsqueeze(0) for x in max1]
    #     print("Time taken for loading: ", time.monotonic() - st )
    #     decoded_kv1, decoded_kv2, decoded_kv3, delay = self.decode_from_pt("/L3C-PyTorch/src/layer1.pt", \
    #         "/L3C-PyTorch/src/layer2.pt", "/L3C-PyTorch/src/layer3.pt", max1)
    #     # # maxes = []

        
    #     for config in self.cachegen_configs:
    #         if not config.is_kv:
    #             st = time.monotonic()
    #             end_index = config.end_index
    #             attention_mask = torch.ones_like(input_ids)
    #             generated = self.model.generate(inputs=input_ids[:, :end_index], 
    #                                             attention_mask=attention_mask[:, :end_index], 
    #                                             past_key_values=prefill_kv,
    #                                             return_dict_in_generate=True,
    #                                             max_length = 0)
    #             del prefill_kv
    #             prefill_kv = generated.past_key_values 
    #             torch.cuda.synchronize()
    #             print("Time taken for generating: ", time.monotonic() - st)
    #         else:
    #             st = time.monotonic()
    #             kv_cache = self.decode( cdf, cdf2, cdf3, input_bits, input_bits2, input_bits3, decoded_kv1, decoded_kv2, decoded_kv3)
    #             end = time.monotonic()
    #             print("Time taken for decoding: ", end-st)
    #             prefill_kv = merge_kv(prefill_kv, kv_from_end(kv_cache, config.start_index, config.end_index), free_left = True, free_right = True)
  
    #     return prefill_kv
        # pass
    def transmit(self, bw, size):
        """ bw: in Mbps
        size: in MB
        """
        time_to_transmit = size * 8 / bw
        time.sleep(time_to_transmit)
        
        

    def concat_max(self, max1):
        """
        Given a dict of max tensors, concatenate them into a single tensor
        """
        maxes = []
        for i in range(len(max1)):
            maxes.append(max1[i].unsqueeze(0))
        return torch.cat(maxes, dim=0)
    def helper_encode(self, start_layer, end_layer, cachegen_encoder,layer_id, is_key=False):
        """Encode the bitstreams and save them to disk
        """
        if is_key:
            key = "key"
        else:
            key = "value"
        from encoder_uniform import CacheGenEncoder
       
        if is_key:
            ac_encoder = CacheGenEncoder(quantized_kv = cachegen_encoder.delta_tensors, start_layer = start_layer, end_layer = end_layer, is_key = is_key)
            ac_encoder.compute_cdf()
            pickle.dump(cachegen_encoder.max_tensors, open(f"bits/max_layer{layer_id}_{key}.pkl", "wb"))
            encoded_streams = ac_encoder.encode(ac_encoder.whole_input_k, ac_encoder.final_cdf)
        else:
            ac_encoder = CacheGenEncoder(quantized_kv = cachegen_encoder.delta_tensors_v, start_layer = start_layer, end_layer = end_layer, is_key = is_key)
            ac_encoder.compute_cdf()
            pickle.dump(cachegen_encoder.max_tensors_v, open(f"bits/max_layer{layer_id}_{key}.pkl", "wb"))
            encoded_streams = ac_encoder.encode(ac_encoder.whole_input_k, ac_encoder.final_cdf)
        pickle.dump(encoded_streams, open(f"bits/bitstreams_layer{layer_id}_{key}.pkl", "wb"))
        pickle.dump(ac_encoder.normalized_cdf, open(f"bits/normalized_cdf_layer{layer_id}_{key}.pkl", "wb"))
        torch.save(ac_encoder.whole_input_k, f"bits/sanity_layer{layer_id}_{key}.pt")
    def decoder_helper(self, config, bitstreams_k, bitstreams_v, bitstreams_k2, bitstreams_v2, bitstreams_k3):
        decoded_k1 = torchac.test(self.cdf_k, bitstreams_k, \
            config.end_index-config.start_index, 40, 100)
        decoded_k2 = torchac.test(self.cdf_k2, bitstreams_k2, \
            config.end_index-config.start_index, 40, 100)
        decoded_k3 = torchac.test(self.cdf_k3, bitstreams_k3, \
            config.end_index-config.start_index, 40, 100)
        decoded_v1 = torchac.test(self.cdf_v, bitstreams_v, \
            config.end_index-config.start_index, 40, 100)
        decoded_v2 = torchac.test(self.cdf_v2, bitstreams_v2, \
            config.end_index-config.start_index, 40, 100)
        decoded_k1 = decoded_k1.reshape((decoded_k1.shape[0], 10, 4096)) - (64 // 2 - 1)
        decoded_k2 = decoded_k2.reshape((decoded_k2.shape[0], 10, 4096)) - (32 // 2 - 1)
        decoded_k3 = decoded_k3.reshape((decoded_k3.shape[0], 12, 4096)) - (16 // 2 - 1)
        decoded_k = torch.cat((decoded_k1, decoded_k2, decoded_k3), dim=1).permute((1, 0, 2)).half()
        for i in range(decoded_k.shape[0]):
            if i < 10:
                os.environ['BINS'] = "64"
            elif i < 20:
                os.environ['BINS'] = "32"
            else:
                os.environ['BINS'] = "16"
            decoded_k[i] = vectorwise_quant(decoded_k[i], self.max_k1[i][config.start_index:config.end_index]).clone()
        decoded_v1 = decoded_v1.reshape((decoded_v1.shape[0], 2, 4096)) - (128 // 2 - 1)
        decoded_v2 = decoded_v2.reshape((decoded_v2.shape[0], 30, 4096)) - (16 // 2 - 1)
        decoded_v = torch.cat((decoded_v1, decoded_v2 ), dim=1).permute((1, 0, 2)).half()
        for i in range(decoded_v.shape[0]):
            if i < 2:
                os.environ['BINS'] = "128"
            else:
                os.environ['BINS'] = "16"
            decoded_v[i] = vectorwise_quant(decoded_v[i], self.max_v1[i][config.start_index:config.end_index])
        transformed_kv = self.transformer_kv_to_tuple(decoded_k, decoded_v)
        return transformed_kv
        
    def get(self, input_ids: torch.Tensor):
        """
        Get the KV cache from the input ids
        Will be called by CacheGenController.get()
        """
        
        bitstreams_k = list(pickle.load(open("bits/bitstreams_layer0_key.pkl", "rb")).values())
        bitstreams_v = list(pickle.load(open("bits/bitstreams_layer0_value.pkl", "rb")).values())
        bitstreams_k2 = list(pickle.load(open("bits/bitstreams_layer1_key.pkl", "rb")).values())
        bitstreams_v2 = list(pickle.load(open("bits/bitstreams_layer1_value.pkl", "rb")).values())
        bitstreams_k3 = list(pickle.load(open("bits/bitstreams_layer2_key.pkl", "rb")).values())
        
        KV = None
        for config in self.cachegen_configs:
            
            if config.is_kv:
                st = time.monotonic()
                kv = self.decoder_helper(config, bitstreams_k[config.start_index:config.end_index], \
                    bitstreams_v[config.start_index:config.end_index], \
                        bitstreams_k2[config.start_index:config.end_index], \
                            bitstreams_v2[config.start_index:config.end_index], \
                            bitstreams_k3[config.start_index:config.end_index])
                print("Time taken for decoding: ", time.monotonic() - st)
                if KV is None:
                    KV = kv
                else:
                    KV = merge_kv(KV, kv)
        return KV
            
    def set_config(self, input_ids):  
        final_loc = len(input_ids[0]) - 1
        self.cachegen_configs = [CacheGenConfig(0,    4000,      True ),
                CacheGenConfig(4000, 8000,      False),
                CacheGenConfig(8000, final_loc, False),
               ]
    
        
    def set(self, kv: KVCache, encode = False):
        """
        Set the model-generated KV cache for a given input ids
        This interface is for potential CacheGen functionalities
        """
        # Quantize first 
        from encoder_uniform import CacheGenEncoder
        cachegen_encoder = CacheGenEncoder(kv_cache=kv)
        cachegen_encoder.naive_quantize()
        # self.helper_encode(0, 10, cachegen_encoder, 0, True)
        
        # self.helper_encode(10, 20, cachegen_encoder, 1, True)
        # self.helper_encode(20, 32, cachegen_encoder, 2, True)
        self.helper_encode(0, 2, cachegen_encoder, 0, False)
        self.helper_encode(2, 32, cachegen_encoder, 1, False)
        # else:
        #     encoded_streams = pickle.load(open("bits/bitstreams_layer1.pkl", "rb"))
        #     normalized_cdf = pickle.load(open("bits/normalized_cdf_layer1.pkl", "rb"))
        #     max_tensors = pickle.load(open("bits/max_layer1.pkl", "rb"))
        #     max_tensors = self.concat_max(max_tensors)
        #     decoded_kvs = []
        #     all_bitstreams = []
        #     for i in range(max_tensors.shape[1]):
        #         # breakpoint()
        #         all_bitstreams += [encoded_streams[i]]
                
        #     out = torchac.test(normalized_cdf, all_bitstreams, len(all_bitstreams), 1, len(all_bitstreams))
        #     out -= (64 // 2 - 1)
        #     # vectorwise_quant(torch.cat(decoded_kvs, dim=0)[:, 0, ], max_tensors.permute(1, 0, 2)[:, :10])
        #     breakpoint()
       
    # def set(self, input_ids):
    #     """
    #     Set the model-generated KV cache for a given input ids

    #     Note:
    #         It's possible that the input_ids is already in the cache gen engine. In this case, we should skip (or maybe overwrite?)
    #     """
    #     # TODO: maybe do the quantization and compression here?
    #     loc1, loc2 = 4000, 4500 # [100, 200, 300, 400, 500, 600, 700, 800]
    #     final_loc = min(len(input_ids[0]) - 1, self.past_kv[0][0].shape[-2])
        
    #     kv1, kv2 = split_kv(self.past_kv, loc1)
    #     _, kv2 = split_kv(self.past_kv, loc2)
    #     kv2, _ = split_kv(kv2, final_loc - loc2)
    #     self.cachegen_configs = [CacheGenConfig(0,    1500,      True ),
    #            CacheGenConfig(1500, 3000,      True),
    #            CacheGenConfig(3000, 4500, True),
    #            CacheGenConfig(4500, 6000, False),
    #            CacheGenConfig(6000, 7500, False),
    #            CacheGenConfig(7500, 9000, True),
    #         #    CacheGenConfig(6000, 7000, True),
    #         #     CacheGenConfig(7000, 8000, True),
    #         #     CacheGenConfig(8000, 9000, True),
    #             CacheGenConfig(9000, final_loc, True),
    #            ]
        
    def transform_tuple_to_kv(self, key):
        k_list = []
        v_list = []
        for i in range(len(key)):
            # tmp = key[i].reshape((key.shape[1], key.shape[2], 128)).permute((1, 0, 2)).unsqueeze(0)
            k_list += [key[i][0].permute((0, 2, 1, 3)).reshape((key[i][0].shape[0], key[i][0].shape[2], -1))]
            v_list += [key[i][1].permute((0, 2, 1, 3)).reshape((key[i][1].shape[0], key[i][1].shape[2], -1))]
        return torch.cat(k_list, dim=0), torch.cat(v_list, dim=0)
    

    def initialize_kv_cache(self, input_ids) -> KVCache:
        """
        Initialize the kv cache for a given input ids
        """
        with torch.no_grad():
            output = self.model.generate(input_ids, 
                       do_sample=False, 
                       max_length=input_ids.shape[-1] + 1,
                       return_dict_in_generate=True,
                       )
        self.past_kv = output["past_key_values"]
        return output["past_key_values"]
        
        


class CacheGenController:
    """
    The cachegen controller
    Currently designed as a singleton class, because we do not want to load the file multiple times. Calling
    `CacheGenController.GetInstance()` will get the singleton object. This behavior could be changed in the future.
    """
    _instances = {}

    # def __init__(self, *args, **kwargs):
    #     self.engine = CacheGenEngine(*args, **kwargs)

    def get(self, input_ids: torch.Tensor) -> List[CacheGenConfig]:
        """
        Get the list of CacheGenConfig from the input ids
        Input:
            input_ids: a torch.Tensor whose shape of N elements. We assume that there is NO batching dimension

        Returns:
            A list of `CacheGenConfig` object, containing the kv cache or tokenized inputs.
            The elements in the list should be sorted by `start_index`.
            For any i, we should have `result[i].end_index == result[i+1].start_index`
            If result[-1] is a kv cache, we should make sure that result[-1].end_index == len(input_ids) - 1

        NOTE:
        - Remember to remove the batching dimension when calling this function
        - Make sure the returned list satisifies the requirements
        """
        return self.engine.get(input_ids)

    
    def set(self, input_ids: torch.Tensor, kv: KVCache):
        """
        Set the model-generated KV cache for a given input ids
        This interface is for potential CacheGen functionalities
        """
        self.engine.set(input_ids, kv)  

    @classmethod
    def GetInstance(cls, *args, **kwargs):
        """
        Return the singleton instance of the CacheGen controller
        """
        if cls not in cls._instances:
            instance = cls(*args, **kwargs)

            cls._instances[cls] = instance

        return cls._instances[cls]

###########
# Helper functions
###########

def merge_kv(left: KVCache, right: KVCache, free_left = False, free_right = False) -> KVCache:
    """
    Merges two kv caches, returns a merged KV cache
    A single KVCache is a tuple_32(tuple_2(torch.Tensor[bs, channels?, num_tokens, hidden_size]))

    Input:
    - left: the left kv cache, could be None
    - right: the right kv cache

    Returns: The merged kv cache. If left is None, returns right
    """
    if left is None:
        return right
    #assert len(left) == len(right)

    def generator():
        for left_layer, right_layer in zip(left, right):
            yield (torch.cat([left_layer[0], right_layer[0]], dim = -2), torch.cat([left_layer[1], right_layer[1]], dim = -2))
            if free_left:
                del left_layer
            if free_right:
                del right_layer

    return tuple(generator())

def split_kv(kv: KVCache, split_index: int) -> Tuple[KVCache, KVCache]:
    """
    Splits a kv cache into two kv caches
    A single KVCache is a tuple_32(tuple_2(torch.Tensor[bs, channels?, num_tokens, hidden_size]))

    Input:
    - kv: the kv cache to be splitted
    - split_index: the index to split the kv cache

    Returns: a tuple of two kv caches
    """
    def generator_left():
        for layer in kv:
            yield (layer[0][:, :, :split_index], layer[1][:, :, :split_index])
    left = tuple(generator_left())
    def generator_right():
        for layer in kv:
            yield (layer[0][:, :, split_index:], layer[1][:, :, split_index:])
    right = tuple(generator_right())
    return left, right

def kv_from_end(kv: KVCache, start_index: int, end_index: int):
    def generator():
        for layer in kv:
            yield (layer[0][:, :, start_index:end_index], layer[1][:, :, start_index:end_index])
    return tuple(generator())

if __name__ == "__main__":
    ## Test the merge_kv function
    def generate_kvcache(shape) -> KVCache:
        """
        Generates a kv cache with the following shape 
        """
        len = 32
        for i in range(len):
            yield (torch.randn(shape), torch.randn(shape))
    kv1 = tuple(generate_kvcache((1, 32, 500, 128)))
    kv2 = tuple(generate_kvcache((1, 32, 800, 128)))
    merged = merge_kv(kv1, kv2)
    print(merged[0][0].shape)
