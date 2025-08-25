using System.Collections;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using UnityEditor;
using UnityEditor.SceneManagement;

public class SolarSystemTest
{
    private const float tolerance = 1e-3f;

    private SolarSystemController controller;
    private Transform sun;
    private Transform earth;
    private Transform moon;

    [SetUp]
    public void Setup()
    {
        EditorSceneManager.OpenScene("Assets/Scenes/SolarSystem.unity");
        controller = GameObject.Find("Solar System Root").GetComponent<SolarSystemController>();
        sun = GameObject.Find("Sun").transform;
        earth = GameObject.Find("Earth").transform;
        moon = GameObject.Find("Moon").transform;
    }

    /*
    /// <summary>
    /// This is just a sanity check.
    /// No points will be granted for this test.
    /// </summary>
    [Test]
    public void ValidScene()
    {
        Assert.That(controller, Is.Not.Null);
        Assert.That(sun, Is.Not.Null);
        Assert.That(earth, Is.Not.Null);
        Assert.That(moon, Is.Not.Null);
    }
    */

    [Test]
    public void TestSun()
    {
        controller.UpdateSolarSystem(0);

        Assert.That(Vector3.Distance(sun.position, Vector3.zero), Is.LessThanOrEqualTo(tolerance));
        Assert.That(Vector3.Distance(sun.rotation.eulerAngles, Vector3.zero), Is.LessThanOrEqualTo(tolerance));
        Assert.That(Vector3.Distance(sun.localScale, new Vector3(1.0f, 1.0f, 1.0f)), Is.LessThanOrEqualTo(tolerance));

        controller.UpdateSolarSystem(123.0f);

        Assert.That(Vector3.Distance(sun.position, Vector3.zero), Is.LessThanOrEqualTo(tolerance));
        Assert.That(Vector3.Distance(sun.rotation.eulerAngles, new Vector3(0.0f, 178.022f, 0.0f)), Is.LessThanOrEqualTo(tolerance));
        Assert.That(Vector3.Distance(sun.localScale, new Vector3(1.0f, 1.0f, 1.0f)), Is.LessThanOrEqualTo(tolerance));

        controller.UpdateSolarSystem(1234.0f);

        Assert.That(Vector3.Distance(sun.position, Vector3.zero), Is.LessThanOrEqualTo(tolerance));
        Assert.That(Vector3.Distance(sun.rotation.eulerAngles, new Vector3(0.0f, 287.4723f, 0.0f)), Is.LessThanOrEqualTo(tolerance));
        Assert.That(Vector3.Distance(sun.localScale, new Vector3(1.0f, 1.0f, 1.0f)), Is.LessThanOrEqualTo(tolerance));
    }

    [Test]
    public void TestEarth()
    {
        controller.UpdateSolarSystem(0);

        Assert.That(Vector3.Distance(earth.position, new Vector3(10.0f, 0.0f, 0.0f)), Is.LessThanOrEqualTo(tolerance));
        Assert.That(Vector3.Distance(earth.rotation.eulerAngles, Vector3.zero), Is.LessThanOrEqualTo(tolerance));
        Assert.That(Vector3.Distance(earth.localScale, new Vector3(0.5f, 0.5f, 0.5f)), Is.LessThanOrEqualTo(tolerance));

        controller.UpdateSolarSystem(123.0f);

        Assert.That(Vector3.Distance(earth.position, new Vector3(-5.1974f, 0.0f, 8.5432f)), Is.LessThanOrEqualTo(tolerance));
        Assert.That(Vector3.Distance(earth.rotation.eulerAngles, new Vector3(0.0f, 238.6852f, 0.0f)), Is.LessThanOrEqualTo(tolerance));
        Assert.That(Vector3.Distance(earth.localScale, new Vector3(0.5f, 0.5f, 0.5f)), Is.LessThanOrEqualTo(tolerance));

        controller.UpdateSolarSystem(1234.0f);

        Assert.That(Vector3.Distance(earth.position, new Vector3(-7.3249f, 0.0f, 6.8077f)), Is.LessThanOrEqualTo(tolerance));
        Assert.That(Vector3.Distance(earth.rotation.eulerAngles, new Vector3(0.0f, 222.9033f, 0.0f)), Is.LessThanOrEqualTo(tolerance));
        Assert.That(Vector3.Distance(earth.localScale, new Vector3(0.5f, 0.5f, 0.5f)), Is.LessThanOrEqualTo(tolerance));
    }

    [Test]
    public void TestMoon()
    {
        controller.UpdateSolarSystem(0);

        Assert.That(Vector3.Distance(moon.position, new Vector3(11.0f, 0.0f, 0.0f)), Is.LessThanOrEqualTo(tolerance));
        Assert.That(Vector3.Distance(moon.rotation.eulerAngles, Vector3.zero), Is.LessThanOrEqualTo(tolerance));
        Assert.That(Vector3.Distance(moon.localScale, new Vector3(0.1f, 0.1f, 0.1f)), Is.LessThanOrEqualTo(tolerance));

        controller.UpdateSolarSystem(123.0f);

        Assert.That(Vector3.Distance(moon.position, new Vector3(-4.6485f, 0.0f, 7.7073f)), Is.LessThanOrEqualTo(tolerance));
        Assert.That(Vector3.Distance(moon.rotation.eulerAngles, new Vector3(0.0f, 56.7069f, 0.0f)), Is.LessThanOrEqualTo(tolerance));
        Assert.That(Vector3.Distance(moon.localScale, new Vector3(0.1f, 0.1f, 0.1f)), Is.LessThanOrEqualTo(tolerance));

        controller.UpdateSolarSystem(1234.0f);

        Assert.That(Vector3.Distance(moon.position, new Vector3(-8.1942f, 0.0f, 6.3134f)), Is.LessThanOrEqualTo(tolerance));
        Assert.That(Vector3.Distance(moon.rotation.eulerAngles, new Vector3(0.0f, 150.3764f, 0.0f)), Is.LessThanOrEqualTo(tolerance));
        Assert.That(Vector3.Distance(moon.localScale, new Vector3(0.1f, 0.1f, 0.1f)), Is.LessThanOrEqualTo(tolerance));
    }

    [TearDown]
    public void Teadown()
    {
        EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);
    }
}
