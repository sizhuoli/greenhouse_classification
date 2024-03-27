import shutil
import os
import subprocess


class WandbUrls:
    def __init__(self, url):

        hash = url.split("/")[-2]
        project = url.split("/")[-3]
        entity = url.split("/")[-4]

        self.weight_url = url
        self.log_url = "https://app.wandb.ai/{}/{}/runs/{}/logs".format(entity, project, hash)
        self.chart_url = "https://app.wandb.ai/{}/{}/runs/{}".format(entity, project, hash)
        self.overview_url = "https://app.wandb.ai/{}/{}/runs/{}/overview".format(entity, project, hash)
        self.hydra_config_url = "https://app.wandb.ai/{}/{}/runs/{}/files/hydra-config.yaml".format(
            entity, project, hash
        )
        self.overrides_url = "https://app.wandb.ai/{}/{}/runs/{}/files/overrides.yaml".format(entity, project, hash)

    def __repr__(self):
        msg = "=================================================== WANDB URLS ===================================================================\n"
        for k, v in self.__dict__.items():
            msg += "{}: {}\n".format(k.upper(), v)
        msg += "=================================================================================================================================\n"
        return msg


class Wandb:
    IS_ACTIVE = False

    @staticmethod
    def set_urls_to_model(model, url):
        wandb_urls = WandbUrls(url)
        model.wandb = wandb_urls

    

    @staticmethod
    def launch(cfg, launch: bool):
        if launch:
            import wandb

            Wandb.IS_ACTIVE = True

            tags = [
                cfg.model_type,
                
            ]

            wandb_args = {}
            wandb_args["project"] = cfg.wandb.project # variable in the config file
            wandb_args["tags"] = tags
            wandb_args["resume"] = "allow"
            wandb_args["name"] = cfg.wandb.name
            

            try:
                commit_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
                gitdiff = subprocess.check_output(["git", "diff", "--", "':!notebooks'"]).decode()
            except:
                commit_sha = "n/a"
                gitdiff = ""

            config = wandb_args.get("config", {})
            wandb_args["config"] = {
                **config,
                "run_path": os.getcwd(),
                "commit": commit_sha,
            }

            wandb.init(**wandb_args)

            filename = os.path.join(os.getcwd(), "./conf/unet_class.yaml")
            ff = os.path.basename(filename)
            
            shutil.copyfile(filename, 
                            os.path.join(wandb.run.dir, ff))
            wandb.save(os.path.join(os.getcwd(), "./unet_class.yaml"))
            

            

    @staticmethod
    def add_file(file_path: str):
        if not Wandb.IS_ACTIVE:
            raise RuntimeError("wandb is inactive, please launch first.")
        import wandb

        filename = os.path.basename(file_path)
        shutil.copyfile(file_path, os.path.join(wandb.run.dir, filename))
