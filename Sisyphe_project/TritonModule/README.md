## Sisyphe 6dof inference triton server

Full inference code of Sisyphe project encapsulation by triton server, including guide and grad pose estimation visualization results.

Main server interface encapsulated in model.py. See details in **execute** method.

This repo minimize the inference code. Potential bugs caused by overwrite or shallow copy in p02\_6dof.py and p10\_infer class Pose\_estimation. If discover and fixed it, contact haoran.chen@micro-i.com.cn and create a pull request. Or request a maintainer permission.

### Constructions

Tree as below

    └── 1
        ├── models
        ├── parameter
        ├── result
        └── utils_place


Dir 'models'  include some tmp visualization files. 'parameter' include config files, such as camera intrinsic file and model file.

In p00\_prep.py, math converter has been encapsulated. 
6dof core functions set in p02\_6dof.py.

Post\-process functions and visualization tools are implemented in p03 and p05 files. See details respectively.

### Scripts

For testing, 

1. use `python t00_infer.py` to get the 6dof result.
2. use `python p10_infer.py` to get the visualization result.

Now the transform presentation is not same in t00 and p10, new version will fix it. If you want to debug the full workflow, try to read and debug the p10\_infer.py.

### Start grpc server

```
bash run.sh
```

### Appendix

Usually, tcp/base/paw coordinate system should be transform in the workflow.

- Transform

    Transform^{a}_{b} means coordinate system b in coordinate system a, or a to b. 

    For example, T^{world}_{cam} means camera pose in the world/base.

- paw: means paw/(maybe jaw is right) along the grasp end. The relation between paw and camera are fixed by hand-eye matrix. Using eye in hand system in Sisyphe.


