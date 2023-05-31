from pytorch3d.loss.chamfer import *
from pytorch3d.loss.chamfer import _handle_pointcloud_input
from pytorch3d.loss.chamfer import _validate_chamfer_reduction_inputs
from pytorch3d.loss import point_mesh_face_distance
from pytorch3d.structures import Meshes, Pointclouds


def scan_to_mesh_face_distance(scan_points: torch.tensor, mesh: Meshes):
    pcd = Pointclouds(points=[scan_points[0]])
    loss = point_mesh_face_distance(mesh, pcd)
    return loss

# Custom chamfer distance:
def scan_to_mesh_distance(scans_points,
                          scans_normals,
                          meshes_points,
                          meshes_normals):

    return custom_chamfer_distance_single_direction(
        scans_points,
        meshes_points,
        x_normals=scans_normals,
        y_normals=meshes_normals,
        threshold=0.00005
    )


def custom_chamfer_distance_single_direction(
        x,
        y,
        x_lengths=None,
        y_lengths=None,
        x_normals=None,
        y_normals=None,
        weights=None,
        batch_reduction: Union[str, None] = "mean",
        point_reduction: str = "mean",
        threshold: float = 0.00001,
        norm: int = 2,
):
    """
    Single direction Chamfer distance from point cloud x to y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
        norm: int indicates the norm used for the distance. Supports 1 for L1 and 2 for L2.

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    x_mask = (
            torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return (x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0

    cham_norm_x = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)

    # mask to filter out those distance pairs that are above the threshold distance
    x_distance_mask = cham_x > threshold
    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0

    cham_x[x_distance_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)

    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]

        cham_norm_x = 1 - torch.abs(
            F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        )

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0

        cham_norm_x[x_distance_mask] = 0.0

        if weights is not None:
            cham_norm_x *= weights.view(N, 1)

    # Apply point reduction
    cham_x = cham_x.sum(1)  # (N,)
    if return_normals:
        cham_norm_x = cham_norm_x.sum(1)  # (N,)
    if point_reduction == "mean":
        x_lengths_distance_masked = max((x_distance_mask == False).sum(-1), 1)
        cham_x /= x_lengths_distance_masked
        if return_normals:
            cham_norm_x /= x_lengths_distance_masked

    if batch_reduction is not None:
        # batch_reduction == "sum"
        cham_x = cham_x.sum()
        if return_normals:
            cham_norm_x = cham_norm_x.sum()
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else max(N, 1)
            cham_x /= div
            if return_normals:
                cham_norm_x /= div

    cham_dist = cham_x
    cham_normals = cham_norm_x if return_normals else None

    return cham_dist, cham_normals