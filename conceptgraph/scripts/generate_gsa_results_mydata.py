from generate_gsa_results import *
import re

def get_frame_number(filename):
    match = re.search(r"frame_(\d+)\.jpg", filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return -1


def main(args: argparse.Namespace):
    ### Initialize the Grounding DINO model ###
    grounding_dino_model = Model(
        model_config_path=GROUNDING_DINO_CONFIG_PATH, 
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, 
        device=args.device
    )

    ### Initialize the SAM model ###
    if args.class_set == "none":
        mask_generator = get_sam_mask_generator(args.sam_variant, args.device)
    else:
        sam_predictor = get_sam_predictor(args.sam_variant, args.device)
    
    ###
    # Initialize the CLIP model
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", "laion2b_s32b_b79k"
    )
    clip_model = clip_model.to(args.device)
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
    
    # Initialize the dataset
    files = [f for f in os.listdir(args.dataset_root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    cx_dataset = sorted(files, key=get_frame_number)

    global_classes = set()
    
    # Initialize a YOLO-World model
    if args.detector == "yolo":
        from ultralytics import YOLO
        yolo_model_w_classes = YOLO('yolov8l-world.pt')  # or choose yolov8m/l-world.pt
    
    if args.class_set == "scene":
        # Load the object meta information
        obj_meta_path = args.dataset_root / args.scene_id / "obj_meta.json"
        with open(obj_meta_path, "r") as f:
            obj_meta = json.load(f)
        # Get a list of object classes in the scene
        classes = process_ai2thor_classes(
            [obj["objectType"] for obj in obj_meta],
            add_classes=[],
            remove_classes=['wall', 'floor', 'room', 'ceiling']
        )
    elif args.class_set == "generic":
        classes = FOREGROUND_GENERIC_CLASSES
    elif args.class_set == "minimal":
        classes = FOREGROUND_MINIMAL_CLASSES
    elif args.class_set in ["tag2text", "ram"]:
        ### Initialize the Tag2Text or RAM model ###
        
        if args.class_set == "tag2text":
            # The class set will be computed by tag2text on each image
            # filter out attributes and action categories which are difficult to grounding
            delete_tag_index = []
            for i in range(3012, 3429):
                delete_tag_index.append(i)

            specified_tags='None'
            # load model
            tagging_model = tag2text.tag2text_caption(pretrained=TAG2TEXT_CHECKPOINT_PATH,
                                                    image_size=384,
                                                    vit='swin_b',
                                                    delete_tag_index=delete_tag_index)
            # threshold for tagging
            # we reduce the threshold to obtain more tags
            tagging_model.threshold = 0.64 
        elif args.class_set == "ram":
            tagging_model = ram(pretrained=RAM_CHECKPOINT_PATH,
                                         image_size=384,
                                         vit='swin_l')
            
        tagging_model = tagging_model.eval().to(args.device)
        
        # initialize Tag2Text
        tagging_transform = TS.Compose([
            TS.Resize((384, 384)),
            TS.ToTensor(), 
            TS.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])
        
        classes = None
    elif args.class_set == "none":
        classes = ['item']
    else:
        raise ValueError("Unknown args.class_set: ", args.class_set)

    if args.class_set not in ["ram", "tag2text"]:
        print("There are total", len(classes), "classes to detect. ")
    elif args.class_set == "none":
        print("Skipping tagging and detection models. ")
    else:
        print(f"{args.class_set} will be used to detect classes. ")
        
    save_name = f"{args.class_set}"
    if args.sam_variant != "sam": # For backward compatibility
        save_name += f"_{args.sam_variant}"
    if args.exp_suffix:
        save_name += f"_{args.exp_suffix}"
    
    if args.save_video:
        # video_save_path = Path(os.path.join("/scratch3/kat049/concept-graphs/my_local_data/DARPA", f"gsa_vis_{save_name}.mp4" ))
        video_save_path = Path(os.path.join(args.dataset_root, f"gsa_vis_{save_name}.mp4" ))
        frames = []
    
    for idx in trange(len(cx_dataset)):
        ### Relevant paths and load image ###
        color_path = os.path.join(args.dataset_root, cx_dataset[idx])

        color_path = Path(color_path)
        
        # vis_save_path = Path(os.path.join("/scratch3/kat049/concept-graphs/my_local_data/DARPA", f"gsa_vis_{save_name}" , color_path.name))
        # detections_save_path =  Path(os.path.join("/scratch3/kat049/concept-graphs/my_local_data/DARPA", f"gsa_detections_{save_name}" , color_path.name))
        vis_save_path = Path(os.path.join(args.dataset_root, f"gsa_vis_{save_name}" , color_path.name))
        detections_save_path =  Path(os.path.join(args.dataset_root, f"gsa_detections_{save_name}" , color_path.name))
        detections_save_path = detections_save_path.with_suffix(".pkl.gz")
        
        os.makedirs(os.path.dirname(vis_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(detections_save_path), exist_ok=True)
        
        # opencv can't read Path objects... sigh...
        color_path = str(color_path)
        vis_save_path = str(vis_save_path)
        detections_save_path = str(detections_save_path)
        
        image = cv2.imread(color_path) # This will in BGR color space
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB color space
        image_pil = Image.fromarray(image_rgb)
        
        ### Tag2Text ###
        if args.class_set in ["ram", "tag2text"]:
            raw_image = image_pil.resize((384, 384))
            raw_image = tagging_transform(raw_image).unsqueeze(0).to(args.device)
            
            if args.class_set == "ram":
                res = inference_ram(raw_image , tagging_model)
                caption="NA"
            elif args.class_set == "tag2text":
                res = inference_tag2text.inference(raw_image , tagging_model, specified_tags)
                caption=res[2]

            # Currently ", " is better for detecting single tags
            # while ". " is a little worse in some case
            text_prompt=res[0].replace(' |', ',')
            # text_prompt = text_prompt + ", spot robot, all terrain robot, red backpack, orange drill, blue barrel, hard white helmet"
            
            # Add "other item" to capture objects not in the tag2text captions. 
            # Remove "xxx room", otherwise it will simply include the entire image
            # Also hide "wall" and "floor" for now...
            add_classes = ["other item"]
            remove_classes = [
                "room", "kitchen", "office", "house", "home", "building", "corner",
                "shadow", "carpet", "photo", "shade", "stall", "space", "aquarium", 
                "apartment", "image", "city", "blue", "skylight", "hallway", 
                "bureau", "modern", "salon", "doorway", "wall lamp", "wood floor"
            ]
            bg_classes = ["wall", "floor", "ceiling"]

            if args.add_bg_classes:
                add_classes += bg_classes
            else:
                remove_classes += bg_classes

            classes = process_tag_classes(
                text_prompt, 
                add_classes = add_classes,
                remove_classes = remove_classes,
            )
            
        # add classes to global classes
        global_classes.update(classes)
        
        if args.accumu_classes:
            # Use all the classes that have been seen so far
            classes = list(global_classes)
            
        ### Detection and segmentation ###
        if args.class_set == "none":
            # Directly use SAM in dense sampling mode to get segmentation
            mask, xyxy, conf = get_sam_segmentation_dense(
                args.sam_variant, mask_generator, image_rgb)
            detections = sv.Detections(
                xyxy=xyxy,
                confidence=conf,
                class_id=np.zeros_like(conf).astype(int),
                mask=mask,
            )
            image_crops, image_feats, text_feats = compute_clip_features(
                image_rgb, detections, clip_model, clip_preprocess, clip_tokenizer, classes, args.device)

            ### Visualize results ###
            annotated_image, labels = vis_result_fast(
                image, detections, classes, instance_random_color=True)
            
            cv2.imwrite(vis_save_path, annotated_image)
        else:
            if args.detector == "dino":
                # Using GroundingDINO to detect and SAM to segment
                detections = grounding_dino_model.predict_with_classes(
                    image=image, # This function expects a BGR image...
                    classes=classes,
                    box_threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                )
            
                if len(detections.class_id) > 0:
                    ### Non-maximum suppression ###
                    # print(f"Before NMS: {len(detections.xyxy)} boxes")
                    nms_idx = torchvision.ops.nms(
                        torch.from_numpy(detections.xyxy), 
                        torch.from_numpy(detections.confidence), 
                        args.nms_threshold
                    ).numpy().tolist()
                    # print(f"After NMS: {len(detections.xyxy)} boxes")

                    detections.xyxy = detections.xyxy[nms_idx]
                    detections.confidence = detections.confidence[nms_idx]
                    detections.class_id = detections.class_id[nms_idx]
                    
                    # Somehow some detections will have class_id=-1, remove them
                    valid_idx = detections.class_id != -1
                    detections.xyxy = detections.xyxy[valid_idx]
                    detections.confidence = detections.confidence[valid_idx]
                    detections.class_id = detections.class_id[valid_idx]

                    # # Somehow some detections will have class_id=-None, remove them
                    # valid_idx = [i for i, val in enumerate(detections.class_id) if val is not None]
                    # detections.xyxy = detections.xyxy[valid_idx]
                    # detections.confidence = detections.confidence[valid_idx]
                    # detections.class_id = [detections.class_id[i] for i in valid_idx]
            elif args.detector == "yolo":
                # YOLO 
                # yolo_model.set_classes(classes)
                yolo_model_w_classes.set_classes(classes)
                yolo_results_w_classes = yolo_model_w_classes.predict(color_path)

                yolo_results_w_classes[0].save(vis_save_path[:-4] + "_yolo_out.jpg")
                xyxy_tensor = yolo_results_w_classes[0].boxes.xyxy 
                xyxy_np = xyxy_tensor.cpu().numpy()
                confidences = yolo_results_w_classes[0].boxes.conf.cpu().numpy()
                
                detections = sv.Detections(
                    xyxy=xyxy_np,
                    confidence=confidences,
                    class_id=yolo_results_w_classes[0].boxes.cls.cpu().numpy().astype(int),
                    mask=None,
                )
                
            if len(detections.class_id) > 0:
                
                ### Segment Anything ###
                detections.mask = get_sam_segmentation_from_xyxy(
                    sam_predictor=sam_predictor,
                    image=image_rgb,
                    xyxy=detections.xyxy
                )

                # Compute and save the clip features of detections  
                image_crops, image_feats, text_feats = compute_clip_features(
                    image_rgb, detections, clip_model, clip_preprocess, clip_tokenizer, classes, args.device)
            else:
                image_crops, image_feats, text_feats = [], [], []
            
            ### Visualize results ###
            annotated_image, labels = vis_result_fast(image, detections, classes)
            
            # save the annotated grounded-sam image
            if args.class_set in ["ram", "tag2text"] and args.use_slow_vis:
                annotated_image_caption = vis_result_slow_caption(
                    image_rgb, detections.mask, detections.xyxy, labels, caption, text_prompt)
                Image.fromarray(annotated_image_caption).save(vis_save_path)
            else:
                cv2.imwrite(vis_save_path, annotated_image)
        
        if args.save_video:
            frames.append(annotated_image)
        
        # Convert the detections to a dict. The elements are in np.array
        results = {
            "xyxy": detections.xyxy,
            "confidence": detections.confidence,
            "class_id": detections.class_id,
            "mask": detections.mask,
            "classes": classes,
            "image_crops": image_crops,
            "image_feats": image_feats,
            "text_feats": text_feats,
        }
        
        if args.class_set in ["ram", "tag2text"]:
            results["tagging_caption"] = caption
            results["tagging_text_prompt"] = text_prompt
        
        # save the detections using pickle
        # Here we use gzip to compress the file, which could reduce the file size by 500x
        with gzip.open(detections_save_path, "wb") as f:
            pickle.dump(results, f)
    
    # save global classes
    # with open (Path(os.path.join("/scratch3/kat049/concept-graphs/my_local_data/DARPA", f"gsa_classes_{save_name}.json")) , "w") as f:
    #     json.dump(list(global_classes), f)
    with open (Path(os.path.join(args.dataset_root, f"gsa_classes_{save_name}.json")) , "w") as f:
        json.dump(list(global_classes), f)
            
    if args.save_video:
        imageio.mimsave(video_save_path, frames, fps=10)
        print(f"Video saved to {video_save_path}")
        

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)