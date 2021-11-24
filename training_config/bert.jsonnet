local bert_path = "../../opt/embeddings/bert-base-cased";
local model_type = "pretrained_transformer";

{
  "dataset_reader":{
    "type": "conll2003-reader",
    "tokenizer": {
      "type": model_type,
      "model_name": bert_path 
    },
    "token_indexers": {
      "bert": {
          "type": model_type,
          "model_name": bert_path
      }
    }
  },
  "train_data_path": "data/conll2003/train.txt",
  "validation_data_path": "data/conll2003/test.txt",
  "test_data_path": "data/conll2003/test.txt",
  "model": {
    "type": "bert",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "pretrained_file": "../../opt/embeddings/glove.840B.300d.txt.tar.gz",
            "type": "embedding",
            "embedding_dim": 300,
            "trainable": true 
        }
      }
    },
    "embedding_dropout": 0.25,
    "pre_encode_feedforward": {
        "input_dim": 300,
        "num_layers": 1,
        "hidden_dims": [300],
        "activations": ["relu"],
        "dropout": [0.25]
    },
    "encoder": {
      "type": "lstm",
      "input_size": 300,
      "hidden_size": 300,
      "num_layers": 1,
      "bidirectional": true
    },
    "integrator": {
      "type": "lstm",
      "input_size": 1800,
      "hidden_size": 300,
      "num_layers": 1,
      "bidirectional": true
    },
    "integrator_dropout": 0.1,
    "projection_layer": {
        "input_dim": 600,
        "num_layers": 1,
        "hidden_dims": [600],
        "activations": ["relu"],
        "dropout": [0.2]
    }
  },
  "data_loader": {
      "batch_size" : 200,
      "shuffle": true

    // "batch_sampler": {
      // "type": "bucket",
      // "batch_size" : 100
    // }
  },
  "trainer": {
    "num_epochs": 100,
    "patience": 5,
    "grad_norm": 5.0,
    "validation_metric": "+f1_score",
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    },
    "cuda_device": 1

  },
  // "distributed": {
    // "cuda_devices": [0, 1]
  // }
}
