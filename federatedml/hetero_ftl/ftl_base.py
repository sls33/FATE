from federatedml.nn.homo_nn.nn_model import get_nn_builder
from federatedml.model_base import ModelBase
from federatedml.param.ftl_param import FTLParam
from federatedml.nn.homo_nn.nn_model import NNModel
from federatedml.nn.backend.tf_keras.nn_model import KerasNNModel
from federatedml.util.classify_label_checker import ClassifyLabelChecker
from federatedml.transfer_variable.transfer_class.ftl_transfer_variable_transfer_variable import FTLTransferVariable
from federatedml.hetero_ftl.ftl_dataloder import FTLDataLoader
from federatedml.nn.hetero_nn.backend.tf_keras.data_generator import KerasSequenceDataConverter
from federatedml.nn.hetero_nn.util import random_number_generator
from federatedml.secureprotol import PaillierEncrypt
from federatedml.secureprotol.encrypt_mode import EncryptModeCalculator
from federatedml.util import consts
from federatedml.nn.hetero_nn.backend.paillier_tensor import PaillierTensor
from federatedml.protobuf.generated.ftl_model_param_pb2 import FTLModelParam
from federatedml.protobuf.generated.ftl_model_meta_pb2 import FTLModelMeta, PredictParam, OptimizerParam

from arch.api.utils import log_utils
from arch.api.table.eggroll.table_impl import DTable
import json

LOGGER = log_utils.getLogger()


class FTL(ModelBase):

    def __init__(self):
        super(FTL, self).__init__()

        # input para
        self.nn_define = None
        self.alpha = None
        self.tol = None
        self.n_iter_no_change = None
        self.validation_freqs = None
        self.optimizer = None
        self.intersect_param = None
        self.config_type = None

        self.encrypted_mode_calculator_param = None

        # runtime variable
        self.verbose = True
        self.nn: KerasNNModel = None
        self.nn_builder = None
        self.model_param = FTLParam()
        self.x_shape = None
        self.data_num = 0
        self.overlap_num = 0
        self.transfer_variable = FTLTransferVariable()
        self.data_convertor = KerasSequenceDataConverter()
        self.mode = 'encrypted'
        self.encrypt_calculators = []
        self.encrypter = None
        self.partitions = 10
        self.batch_size = None
        self.epochs = None

    def _init_model(self, param: FTLParam):
        self.nn_define = param.nn_define
        self.alpha = param.alpha
        self.tol = param.tol
        self.n_iter_no_change = param.n_iter_no_change
        self.validation_freqs = param.validation_freqs
        self.optimizer = param.optimizer
        self.intersect_param = param.intersect_param
        self.config_type = param.config_type
        self.l2 = param.l2
        self.batch_size = param.batch_size
        self.epochs = param.epochs

        self.encrypted_mode_calculator_param = param.encrypted_mode_calculator_param
        self.encrypter = self.generate_encrypter(param)
        self.rng_generator = random_number_generator.RandomNumberGenerator()

    @staticmethod
    def debug_data_inst(data_inst):
        collect_data = list(data_inst.collect())
        LOGGER.debug('showing DTable')
        for d in collect_data:
            LOGGER.debug('key {} id {}, features {} label {}'.format(d[0], d[1].inst_id, d[1].features, d[1].label))

    @staticmethod
    def check_label(data_inst: DTable):

        LOGGER.debug('checking label')
        label_checker = ClassifyLabelChecker()
        num_class, class_set = label_checker.validate_label(data_inst)
        if num_class != 2:
            raise ValueError('ftl only support binary classification, however {} labels are provided.'.format(num_class))

        if 1 in class_set and -1 in class_set:
            return data_inst
        else:
            new_label_mapping = {list(class_set)[0]: 1, list(class_set)[1]: -1}

            def reset_label(inst):
                inst.label = new_label_mapping[inst.label]
                return inst

            new_table = data_inst.mapValues(reset_label)
            return new_table

    def generate_encrypter(self, param) -> PaillierEncrypt:
        LOGGER.info("generate encrypter")
        if param.encrypt_param.method.lower() == consts.PAILLIER.lower():
            encrypter = PaillierEncrypt()
            encrypter.generate_key(param.encrypt_param.key_length)
        else:
            raise NotImplementedError("encrypt method not supported yet!!!")

        return encrypter

    def generated_encrypted_calculator(self):
        encrypted_calculator = EncryptModeCalculator(self.encrypter,
                                                     self.encrypted_mode_calculator_param.mode,
                                                     self.encrypted_mode_calculator_param.re_encrypted_rate)

        return encrypted_calculator

    def encrypt_tensor(self, components):

        if len(self.encrypt_calculators) == 0:
            self.encrypt_calculators = [self.generated_encrypted_calculator() for i in range(3)]
        encrypted_tensors = []
        for comp, calculator in zip(components, self.encrypt_calculators):
            encrypted_tensor = PaillierTensor(ori_data=comp, partitions=self.partitions)
            encrypted_tensors.append(encrypted_tensor.encrypt(calculator))

        return encrypted_tensors

    def prepare_data(self, intersect_obj, data_inst: DTable, guest_side=False):

        """
        find intersect ids and prepare dataloader
        """
        overlap_samples = intersect_obj.run(data_inst)  # find intersect ids
        non_overlap_samples = data_inst.subtractByKey(overlap_samples)

        if overlap_samples.count() == 0:
            raise ValueError('no intersect samples')

        LOGGER.debug('has {} overlap samples'.format(overlap_samples.count()))

        if guest_side:
            data_inst = self.check_label(data_inst)
        data_loader = FTLDataLoader(non_overlap_samples=non_overlap_samples,
                                    batch_size=self.batch_size, overlap_samples=overlap_samples, guest_side=guest_side)

        LOGGER.debug("data details are :{}".format(data_loader.print_basic_info()))

        self.x_shape = data_loader.x_shape
        self.data_num = data_inst.count()
        self.overlap_num = overlap_samples.count()

        return data_loader, data_loader.x_shape, data_inst.count(), len(data_loader.get_overlap_indexes())

    def initialize_nn(self, input_shape):
        loss = "keep_predict_loss"
        self.nn_builder = get_nn_builder(config_type=self.config_type)
        LOGGER.debug('input shape is {}'.format(input_shape))
        self.nn: NNModel = self.nn_builder(loss=loss, nn_define=self.nn_define, optimizer=self.optimizer, metrics=None,
                                           input_shape=input_shape)
        LOGGER.debug('printing nn layers structure')
        for layer in self.nn._model.layers:
            LOGGER.debug('input shape {}, output shape {}'.format(layer.input_shape, layer.output_shape))

    def generate_mask(self, shape):
        return self.rng_generator.generate_random_number(shape)

    def _batch_gradient_update(self, X, grads):
        data = self.data_convertor.convert_data(X, grads)
        self.nn.train(data)
        LOGGER.debug('optimizer config is {}'.format(self.nn.export_optimizer_config()))

    def _get_mini_batch_gradient(self, X_batch, backward_grads_batch):

        grads = self.nn.get_weight_gradients(X_batch, backward_grads_batch)
        return grads

    def update_nn_weights(self, backward_grads, data_loader: FTLDataLoader, epoch_idx):

        LOGGER.debug('updating grads')

        # self._batch_gradient_update(data_loader.x, backward_grads)

        assert len(data_loader.x) == len(backward_grads)

        weight_grads = []
        for i in range(len(data_loader)):
            start, end = data_loader.get_batch_indexes(i)
            batch_x = data_loader.x[start: end]
            batch_grads = backward_grads[start: end]
            batch_weight_grads = self._get_mini_batch_gradient(batch_x, batch_grads)
            if len(weight_grads) == 0:
                weight_grads.extend(batch_weight_grads)
            else:
                for w, bw in zip(weight_grads, batch_weight_grads):
                    w += bw
        #
        # for i in range(len(weight_grads)):
        #     weight_grads[i] = weight_grads[i]/self.data_num

        # LOGGER.debug('weights grads is {}'.format(weight_grads))

        self.nn.apply_gradients(weight_grads)

        if self.verbose:
            tw = self.nn.get_trainable_weights()
            LOGGER.debug('trainable weights of epoch {} is {}'.format(epoch_idx, tw))

    def export_nn(self):
        return self.nn.export_model()

    def get_model_meta(self):

        model_meta = FTLModelMeta()
        model_meta.config_type = self.config_type
        model_meta.nn_define = json.dumps(self.nn_define)
        model_meta.batch_size = self.batch_size
        model_meta.epochs = self.epochs
        model_meta.tol = self.tol
        model_meta.input_dim = self.x_shape[0]

        predict_param = PredictParam()

        optimizer_param = OptimizerParam()
        optimizer_param.optimizer = self.optimizer.optimizer
        optimizer_param.kwargs = json.dumps(self.optimizer.kwargs)

        model_meta.optimizer_param.CopyFrom(optimizer_param)
        model_meta.predict_param.CopyFrom(predict_param)

        # model_meta.optimizer_param.CopyFrme(optimizer_param)
        return model_meta

    def get_model_param(self):

        model_param = FTLModelParam()
        model_bytes = self.nn.export_model()
        model_param.model_bytes = model_bytes

        return model_param

    def set_model_meta(self, model_meta):

        self.config_type = model_meta.config_type
        self.nn_define = json.loads(model_meta.nn_define)
        self.batch_size = model_meta.batch_size
        self.epochs = model_meta.epochs
        self.tol = model_meta.tol
        self.optimizer = FTLParam()._parse_optimizer(FTLParam().optimizer)
        input_dim = model_meta.input_dim

        self.optimizer.optimizer = model_meta.optimizer_param.optimizer
        self.optimizer.kwargs = json.loads(model_meta.optimizer_param.kwargs)

        self.initialize_nn((input_dim, ))

    def set_model_param(self, model_param):
        self.nn.restore_model(model_param.model_bytes)







