import json
import numpy as np
import cv2
import tqdm
import torch

from eft.utils.imutils import crop, crop_bboxInfo
from eft.utils.imutils import convert_smpl_to_bbox, convert_bbox_to_oriIm, conv_bboxinfo_bboxXYXY
from renderer import viewer2D
from renderer import meshRenderer #screen less opengl renderer
from renderer import glViewer #gui mode opengl renderer
from renderer import denseposeRenderer #densepose renderer
from eft.models import SMPL


def getRenderer(ren_type='geo'):
    """
    Choose renderer type
    geo: phong-shading (silver color)
    colorgeo: phong-shading with color (need color infor. Default silver color)
    denspose: densepose IUV
    normal: normal map
    torch3d: via pytorch3d TODO
    """
    if ren_type=='geo':
        renderer = meshRenderer.meshRenderer()
        renderer.setRenderMode('geo')

    elif ren_type=='colorgeo':
        renderer = meshRenderer.meshRenderer()
        renderer.setRenderMode('colorgeo')
        
    elif ren_type=='normal':
        renderer = meshRenderer.meshRenderer()
        renderer.setRenderMode('normal')

    elif ren_type=='densepose':
        renderer = denseposeRenderer.denseposeRenderer()

    # elif  ren_type=='torch3d':
    #     renderer = torch3dRenderer.torch3dRenderer()
    else:
        assert False

    renderer.offscreenMode(True)
    # renderer.bAntiAliasing= False
    return renderer


def extract_faces(faces, part_vertices = range(0,3000)):
    # vertices 是所有的顶点的列表
    # faces 是所有的面的列表，每个面都是一个顶点的编号的列表
    # part_vertices 是你想要提取的部分的顶点的编号的列表
    
    # 结果集
    result = []
    faces = faces.tolist()
    # 用于记录已经访问过的面
    
    for face in faces:
        # 检查当前面的所有顶点是否属于你想要的部分
        is_part = all(vertex in part_vertices for vertex in face)
        # 如果是，将当前面加入结果集
        if is_part:
            result.append(face)

    return np.array(result)


def generate_mask(smpl_model, part_segm, camera_pose, pose, shape, renderer = None,center=None, scale=None):
    BBOX_IMG_RES = 224
    # part_segm_filepath = "/home/zjt/jieting_ws/MEBOW/smpl_vert_segmentation.json"
    # part_segm = json.load(open(part_segm_filepath))
    # smpl = SMPL('/home/zjt/jieting_ws/eft/data/body_models/smpl',batch_size=1, create_transl=False)

    pred_betas = np.reshape(np.array(shape, dtype=np.float32), (1,10) )     #(10,)
    pred_betas = torch.from_numpy(pred_betas)

    pred_pose_rotmat = np.reshape( np.array(pose, dtype=np.float32), (1,24,3,3)  )        #(24,3,3)
    pred_pose_rotmat = torch.from_numpy(pred_pose_rotmat)

    smpl_output = smpl_model(betas=pred_betas, body_pose=pred_pose_rotmat[:,1:], global_orient=pred_pose_rotmat[:,[0]], pose2rot=False)
    smpl_vertices = smpl_output.vertices.detach().cpu().numpy()[0]
    smpl_joints_3d = smpl_output.joints.detach().cpu().numpy()[0]

    camParam_scale = camera_pose[0]
    camParam_trans = camera_pose[1:]
    pred_vert_vis = smpl_vertices
    # smpl_joints_3d_vis = smpl_joints_3d
    pred_vert_vis = convert_smpl_to_bbox(pred_vert_vis, camParam_scale, camParam_trans)

    renderer.setBackgroundTexture(np.zeros((56,56,3)))

    mask = []

    for part_idx, (k, part_vetices_index) in enumerate(part_segm.items()):
        part_faces = extract_faces(smpl_model.faces, part_vetices_index)
        renderer.set_mesh(pred_vert_vis, part_faces)
        renderer.showBackground(True)

        renderer.setWorldCenterBySceneCenter()

        renderer.setCameraViewMode("cam")

        #Set image size for rendering
        renderer.setViewportSize(56, 56)
        renderImg = renderer.get_screen_color_ibgr()
        # import pdb;pdb.set_trace()
        renderer.show_once()

        ret, mask_all = cv2.threshold(src=renderImg,         
                                    thresh=1,               
                                    maxval=255,               
                                    type=cv2.THRESH_BINARY)
        mask.append(mask_all[:, :, 0])
        cv2.imshow("tes", mask_all)
        cv2.waitKey(1)

    return np.array(mask,dtype=np.float)