// SPDX-License-Identifier: GPL-3.0-or-later
// This executable links PSD2Live's GPL Umamo. It is a separate process from the neutral converter.
package popy

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlinx.serialization.json.*
import org.umamo.runtime.model.*
import org.umamo.interop.moc3.export.Moc3Export
import org.umamo.interop.cmo3.Cmo3Conversion
import org.umamo.interop.cmo3.Cmo3Import
import org.umamo.format.moc3.Moc3
import org.umamo.format.cmo3.Cmo3

private fun JsonObject.s(k:String)=getValue(k).jsonPrimitive.content
private fun JsonObject.f(k:String)=getValue(k).jsonPrimitive.float
private fun JsonObject.a(k:String)=getValue(k).jsonArray
private fun JsonArray.floats()=map{it.jsonPrimitive.float}.toFloatArray()
private fun JsonArray.ints()=map{it.jsonPrimitive.int}.toIntArray()

fun main(args:Array<String>) {
    if(args.contains("--deduplicate-positions")) System.setProperty("popy.moc3.deduplicatePositions","true")
    if(args.firstOrNull()=="--verify-cmo") {
        val file=File(args[1])
        val model=Cmo3Import.fromModelSource(Cmo3.read(file.readBytes()).root as org.umamo.format.cmo3.model.custom.CModelSource)
        val report=buildJsonObject{put("drawables",model.drawables.size);put("parameters",model.parameters.size);put("glues",model.glues.size);put("animated_glues",model.glues.count{it.channelGrids.gridsByChannel.isNotEmpty()});put("masked_drawables",model.drawables.count{it.maskedBy.isNotEmpty()});put("inverted_masks",model.drawables.count{it.invertMask});put("official_editor_verified",false)}
        File(file.parentFile,"cmo_roundtrip.json").writeText(report.toString())
        println("CMO3 own-codec roundtrip: $report")
        return
    }
    require(args.size>=2){"Usage: popy-cubism-exporter IR.json OUTPUT_DIR [--no-cmo]"}
    val input=File(args[0]); val base=input.parentFile; val out=File(args[1]); out.mkdirs()
    var ir=Json.parseToJsonElement(input.readText()).jsonObject
    val absolute=ir["geometry_encoding"]?.jsonPrimitive?.content=="parent_absolute"
    val binary=ByteBuffer.wrap(File(base,ir.s("keyform_data")).readBytes()).order(ByteOrder.LITTLE_ENDIAN)
    fun floats(offset:Int,count:Int):FloatArray=FloatArray(count){binary.getFloat((offset+it)*4)}
    fun axes(o:JsonObject)=o.a("axes").map { e-> val a=e.jsonObject; KeyformAxis(ParameterId(a.s("parameter")),a.a("keys").floats()) }
    val params=ir.a("parameters").map { e->val p=e.jsonObject; Parameter(ParameterId(p.s("id")),p.s("id"),p.f("min"),p.f("max"),p.f("default"),if(p["kind"]?.jsonPrimitive?.content=="blend") ParameterKind.BLEND_SHAPE else ParameterKind.NORMAL) }
    val w=ir.a("canvas")[0].jsonPrimitive.float; val h=ir.a("canvas")[1].jsonPrimitive.float
    val warpId=DeformerId("D_GlobalProjection")
    val headWarpId=DeformerId("D_HeadProjection")
    val bodyRotationId=DeformerId("D_BodyRoll")
    val headRotationId=DeformerId("D_HeadRoll")
    val warp=ir.getValue("root_warp").jsonObject
    val warpGrid=KeyformGrid(axes(warp),warp.a("cells").map {e->val c=e.jsonObject; KeyformCell(c.a("coordinate").ints(),WarpLatticeForm(c.a("points").floats()))})
    val bodyRotation=ir.getValue("body_rotation").jsonObject
    val bodyPivot=bodyRotation.a("pivot").floats()
    val bodyRotGrid=KeyformGrid(listOf(KeyformAxis(ParameterId("ParamBodyAngleZ"),floatArrayOf(-15f,0f,15f))),listOf(-15f,0f,15f).mapIndexed{i,z->KeyformCell(intArrayOf(i),RotationPivotForm(bodyPivot[0],bodyPivot[1],z,1f))})
    val headRotation=ir.getValue("head_rotation").jsonObject
    val headRotGrid=KeyformGrid(axes(headRotation),headRotation.a("cells").map{v->val c=v.jsonObject;val p=c.a("origin").floats();KeyformCell(c.a("coordinate").ints(),RotationPivotForm(p[0],p[1],c.f("angle"),1f))})
    val headWarp=ir.getValue("head_warp").jsonObject
    val headWarpGrid=KeyformGrid(axes(headWarp),headWarp.a("cells").map{v->val c=v.jsonObject;KeyformCell(c.a("coordinate").ints(),WarpLatticeForm(c.a("points").floats()))})
    val bodyWarps=ir["body_warps"]?.jsonArray?.map{v->val node=v.jsonObject
        if(node["kind"]?.jsonPrimitive?.content=="rotation") Deformer.Rotation(DeformerId(node.s("id")),"Native skeleton coordinate carrier",null,null,0f,
            KeyformGrid(emptyList(),listOf(KeyformCell(intArrayOf(),RotationPivotForm(0f,0f,0f,1f)))))
        else Deformer.Warp(DeformerId(node.s("id")),"Skin-weighted pitch correction ${node.f("strength")}",if(node["root"]?.jsonPrimitive?.boolean==true) null else bodyRotationId,null,1,1,true,
        KeyformGrid(axes(node),node.a("cells").map{c->val cell=c.jsonObject;KeyformCell(cell.a("coordinate").ints(),WarpLatticeForm(cell.a("points").floats()))}))} ?: listOf(Deformer.Warp(warpId,"Body projection / breath",bodyRotationId,null,1,1,true,warpGrid))
    val deformers=listOf(
        Deformer.Rotation(bodyRotationId,"Body roll (exact angle)",null,null,0f,bodyRotGrid),
        Deformer.Rotation(headRotationId,"Head roll / body-yaw pivot",bodyRotationId,null,0f,headRotGrid),
        Deformer.Warp(headWarpId,"Head projection / breath",headRotationId,null,1,1,true,headWarpGrid))+bodyWarps
    var forms=0L
    val drawables=ir.a("meshes").mapIndexed {index,e->
        val m=e.jsonObject; val pos=m.a("positions").floats(); val n=pos.size
        val ax=axes(m)
        val geom=KeyformGrid(ax,m.a("cells").map {v->val c=v.jsonObject; forms++; KeyformCell(c.a("coordinate").ints(),MeshDeltaForm(floats(c.getValue("offset").jsonPrimitive.int,n)))})
        val order:KeyformGrid<ChannelValue> = KeyformGrid(ax,m.a("cells").map {v->val c=v.jsonObject; KeyformCell(c.a("coordinate").ints(),ChannelValue.Scalar(c.f("order")))})
        val blends=m["blend_shapes"]?.jsonArray?.map {v->val b=v.jsonObject
            val f=b.a("offsets").mapIndexed {i,off-> if(off is JsonNull) null else MeshForm(floats(off.jsonPrimitive.int,n),drawOrder=b["orders"]?.jsonArray?.get(i)?.jsonPrimitive?.float ?: 500f,opacity=m["opacity"]?.jsonPrimitive?.float ?: 1f)}
            BlendShapeBinding(ParameterId(b.s("parameter")),b.a("keys").floats(),b.getValue("neutral").jsonPrimitive.int,f)
        } ?: emptyList()
        val masks=m["masked_by"]?.jsonArray?.map{DrawableId(it.jsonPrimitive.content)} ?: emptyList()
        val parent=if(m["deformer"]?.jsonPrimitive?.content=="head") headWarpId else m["projection_warp"]?.jsonPrimitive?.content?.let{DeformerId(it)} ?: warpId
        Drawable(DrawableId(m.s("id")),m.s("id"),parent,BlendMode.Normal,masks,DrawableMesh(if(absolute) FloatArray(n) else pos,m.a("uvs").floats(),m.a("indices").ints()),geom,
            channelGrids=ChannelGrids(mapOf(FormChannel.DRAW_ORDER to order)),drawOrder=500f,opacity=m["opacity"]?.jsonPrimitive?.float ?: 1f,texturePage=m.getValue("texture").jsonPrimitive.int,culling=m["culling"]?.jsonPrimitive?.boolean ?: true,blendShapes=blends,
            invertMask=m["invert_mask"]?.jsonPrimitive?.boolean ?: false,
            // Dense 3D correspondence stays in Rig IR. MOC3 does not serialize
            // this editor-side string, so don't retain a second copy at runtime export.
            userData=if(args.contains("--no-cmo")) "" else m.getValue("source").toString())
    }
    val parts=ir.a("parts").map{v->val p=v.jsonObject;Part(PartId(p.s("id")),p.s("id"),p.a("mesh_ids").map{OrgChild.Drawable(DrawableId(it.jsonPrimitive.content))})}
    val glues=ir["glues"]?.jsonArray?.map{v->
        val g=v.jsonObject
        val tracks=if(g.containsKey("axes")) {
            val ax=axes(g)
            val forms:List<KeyformCell<ChannelValue>> = if(g.containsKey("intensities")) g.a("intensities").mapIndexed{i,v->
                var remainder=i
                val coordinate=IntArray(ax.size)
                for(j in ax.indices.reversed()){val size=g.a("axes")[j].jsonObject.a("keys").size;coordinate[j]=remainder%size;remainder/=size}
                KeyformCell(coordinate,ChannelValue.Scalar(v.jsonPrimitive.float))
            } else g.a("cells").map{e->val c=e.jsonObject;KeyformCell(c.a("coordinate").ints(),ChannelValue.Scalar(c.f("intensity")))}
            ChannelGrids(mapOf(FormChannel.GLUE_INTENSITY to KeyformGrid(ax,forms)))
        } else ChannelGrids.Empty
        Glue(DrawableId(g.s("mesh_a")),DrawableId(g.s("mesh_b")),g.a("pairs").map{e->val p=e.jsonArray;GluePair(p[0].jsonPrimitive.int,p[1].jsonPrimitive.int,p[2].jsonPrimitive.float,p[3].jsonPrimitive.float)},channelGrids=tracks,intensity=g["intensity"]?.jsonPrimitive?.float ?: 1f,id=g.s("id"))
    } ?: emptyList()
    val puppet=PuppetModel(params,parts,deformers,drawables,parts.map{OrgChild.Part(it.id)},null,
        renderRoot=RenderGroup(null,500,drawables.map{RenderDrawable(it.id)}),
        glues=glues,
        canvasWidth=w,canvasHeight=h,worldOriginX=w/2,worldOriginY=-h/2,pixelsPerUnit=w/2,runtimeTarget=RuntimeTarget.Cubism50)
    val textures=ir.a("textures").map {it.jsonPrimitive.content}
    // Large geometry/Glue IR trees are no longer needed for runtime-only exports.
    // Release them before lowering, rather than retaining both representations at
    // the peak of the binary writer. CMO3 still needs the authored rest positions.
    if(args.contains("--no-cmo")) ir=JsonObject(emptyMap())
    val lowered=Moc3Export.toMocDocument(puppet)
    val mocBytes=Moc3.write(lowered.document)
    File(out,"POPY.moc3").writeBytes(mocBytes)
    // Independent parser, parameter evaluation and GPU validation run in Ayagami.
    println("MOC3 wrote ${mocBytes.size} bytes, ${drawables.size} meshes, $forms authored keyforms")
    val notices=mutableListOf("MOC3: ${lowered.report}")
    textures.forEach { File(base,it).copyTo(File(out,File(it).name),overwrite=true) }
    val model=buildJsonObject {
        put("Version",3); putJsonObject("FileReferences") {put("Moc","POPY.moc3");putJsonArray("Textures"){textures.forEach{add(File(it).name)}};put("DisplayInfo","POPY.cdi3.json")}
        putJsonArray("Groups") {
            add(buildJsonObject {put("Target","Parameter");put("Name","EyeBlink");putJsonArray("Ids"){add("ParamEyeLOpen");add("ParamEyeROpen")}})
            add(buildJsonObject {put("Target","Parameter");put("Name","LipSync");putJsonArray("Ids"){add("ParamMouthOpenY")}})
        }
    }
    File(out,"POPY.model3.json").writeText(Json{prettyPrint=true}.encodeToString(JsonObject.serializer(),model))
    val cdi=buildJsonObject {put("Version",3);putJsonArray("Parameters"){params.forEach{add(buildJsonObject{put("Id",it.id.raw);put("GroupId","");put("Name",it.name)})}};putJsonArray("ParameterGroups"){};putJsonArray("Parts"){parts.forEach{add(buildJsonObject{put("Id",it.id.raw);put("Name",it.name)})}} }
    File(out,"POPY.cdi3.json").writeText(cdi.toString())
    if(!args.contains("--no-cmo")) {
        // CMO3 source-art placement retains canvas rest positions. MOC3 above uses a zero
        // base and absolute parent coordinates to avoid catastrophic float cancellation.
        val cmoPuppet=if(!absolute) puppet else puppet.copy(drawables=drawables.mapIndexed{i,d->
            val pos=ir.a("meshes")[i].jsonObject.a("positions").floats()
            fun relative(a:FloatArray)=FloatArray(a.size){a[it]-pos[it]}
            d.copy(mesh=DrawableMesh(pos,d.mesh!!.uvs,d.mesh!!.indices),
                geometryGrid=d.geometryGrid?.let{g->KeyformGrid(g.axes,g.cells.map{c->KeyformCell(c.coordinate,MeshDeltaForm(relative(c.form.positionDeltas)))})},
                blendShapes=d.blendShapes.map{b->b.copy(forms=b.forms.map{f->f?.let{MeshForm(relative(it.positionDeltas),it.drawOrder,it.opacity,it.multiplyColor,it.screenColor)}})})
        })
        val pages=textures.map {path->val f=File(base,path);val im=javax.imageio.ImageIO.read(f);Cmo3Conversion.AtlasPage(f.readBytes(),im.width,im.height)}
        val result=Cmo3Conversion.freshCmo3(cmoPuppet,pages,drawables.associate{it.id.raw to it.texturePage},"POPY",0L,0x504f5059)
        val bytes=Cmo3.write(result.model); File(out,"POPY.cmo3").writeBytes(bytes)
        notices.add("CMO3: ${result.report}")
        notices.add("CMO3 written. Run --verify-cmo in a separate JVM to avoid retaining the writer graph during reimport. Official Cubism Editor verification NOT performed.")
        println("CMO3 wrote ${bytes.size} bytes; verify separately with --verify-cmo")
    }
    File(out,"export_report.txt").writeText(notices.joinToString("\n"))
}
