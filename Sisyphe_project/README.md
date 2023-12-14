## Sisyphe code

Server/client code of Sisyphe project.

If discover bugs and fixed, contact haoran.chen@micro-i.com.cn and create a pull request. Or request a maintainer permission. Highly recommend to make a own dev branch and request for merge.

### Constructions

Tree as below

    └── Sisyphe_project
        ├── agilebot_sisyphe (submodule from ise)
        ├── data_processor (scripts for data generation)
        ├── driver (paw controller)
        ├── kpt (submodule for keypoints detection)
        ├── MvImport (camera sdk)
        ├── objective-mtl (submodule for 6dof model training)
        ├── sisiphy_grpc (client for grasp control)
        └── TritonModule (triton server for 6dof inference)
